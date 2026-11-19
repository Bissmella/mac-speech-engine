"""Local STT+TTS WebSocket server for Apple Silicon (MLX).

/stt  Kyutai STT (streaming). Resamples the 16 kHz wire input to 24 kHz, feeds
      80 ms blocks, emits `partial` frames live and one `final` on stop.
/tts  Kokoro TTS (MLX). Synthesizes buffered text on flush/stop and streams PCM.

Localhost only, one connection = one session. See README.

STT uses moshi_mlx 0.2.12. STT and TTS require Apple Silicon with MLX.
"""

import asyncio
import base64
import json
import logging
import math
import os
import queue
import re
import threading
import time

import numpy as np
import websockets

import mlx.core as mx
import mlx.nn as nn
import rustymimi
import sentencepiece
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils

HOST = "127.0.0.1"
PORT = 8765

STT_REPO = "kyutai/stt-1b-en_fr-mlx"          # 1B en+fr, ~0.5s delay + semantic VAD; 2.6b-en is 2.5s

KOKORO_MODEL = "mlx-community/Kokoro-82M-bf16"
KOKORO_VOICE = "af_heart"
KOKORO_SPEED = float(os.environ.get("KOKORO_SPEED", "1.25"))
if not math.isfinite(KOKORO_SPEED) or KOKORO_SPEED <= 0:
    raise ValueError("KOKORO_SPEED must be a finite number greater than zero.")
TTS_MIN_CLAUSE_CHARS = 80
TTS_MAX_BUFFER_CHARS = 160
TTS_SENTENCE_END = re.compile(r"""[.!?]["')\]]*\s+""")
TTS_CLAUSE_END = re.compile(r"[,;:]\s+")

STT_SR = 24000          # Kyutai STT operates at 24 kHz
BLOCK = 1920            # 80 ms at 24 kHz, the frame size the model steps on
FLUSH_BLOCKS = 16       # ~1.3s of trailing silence to drain the 1b model's ~0.5s lookahead; tune per model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("speech")

# MLX isn't safe under concurrent model calls; serialize all inference across threads.
# ponytail: one lock, held per STT utterance / per TTS synth. Fine for localhost half-duplex use.
_mlx_lock = threading.Lock()

_STT = None          # loaded at startup
_kokoro = None       # loaded on first TTS request


# --- helpers -----------------------------------------------------------------

def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x
    n = int(len(x) * dst / src)
    return np.interp(np.linspace(0, len(x), n, endpoint=False), np.arange(len(x)), x).astype(np.float32)


def _audio_frame(samples: np.ndarray, sr: int) -> str:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    return json.dumps({
        "type": "audio",
        "data": base64.b64encode(pcm).decode("ascii"),
        "sample_rate": int(sr),
        "format": "pcm_s16le",
    })


# --- STT (Kyutai) ------------------------------------------------------------

def _load_stt() -> dict:
    cfg_path = hf_hub_download(STT_REPO, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    mimi_weights = hf_hub_download(STT_REPO, cfg["mimi_name"])
    moshi_weights = hf_hub_download(STT_REPO, cfg.get("moshi_name", "model.safetensors"))
    tokenizer = hf_hub_download(STT_REPO, cfg["tokenizer_name"])

    lm_config = models.LmConfig.from_config_dict(cfg)
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    if moshi_weights.endswith(".q4.safetensors"):
        nn.quantize(model, bits=4, group_size=32)
    elif moshi_weights.endswith(".q8.safetensors"):
        nn.quantize(model, bits=8, group_size=64)
    model.load_weights(moshi_weights, strict=True)
    model.warmup()

    return {
        "model": model,
        "text_tok": sentencepiece.SentencePieceProcessor(tokenizer),
        "mimi_weights": mimi_weights,
        "mimi_cb": max(lm_config.generated_codebooks, lm_config.other_codebooks),
        "other_cb": lm_config.other_codebooks,
    }


def _stt_step(stt: dict, gen, mimi, block: np.ndarray) -> str | None:
    """One 1920-sample float32 block -> new text piece (or None). Mirrors stt_from_mic_mlx.py."""
    b = block[None, :]                                  # (1, 1920)
    toks = mimi.encode_step(b[None, 0:1])
    toks = mx.array(toks).transpose(0, 2, 1)[:, :, :stt["other_cb"]]
    tok = gen.step(toks[0])[0].item()
    if tok in (0, 3):                                   # pad / separator tokens
        return None
    return stt["text_tok"].id_to_piece(tok).replace("▁", " ")


async def _stt_session(ws):
    loop = asyncio.get_running_loop()
    in_q: queue.Queue = queue.Queue()      # 24 kHz blocks -> worker thread
    out_q: asyncio.Queue = asyncio.Queue()  # ("partial"|"final", text) -> here

    def worker():
        with _mlx_lock:  # ponytail: held for the whole utterance; half-duplex, so TTS won't contend
            gen = models.LmGen(
                model=_STT["model"], max_steps=4096,
                text_sampler=utils.Sampler(top_k=25, temp=0),
                audio_sampler=utils.Sampler(top_k=250, temp=0.8), check=False,
            )
            mimi = rustymimi.Tokenizer(_STT["mimi_weights"], num_codebooks=_STT["mimi_cb"])
            text = ""
            try:
                while (block := in_q.get()) is not None:
                    piece = _stt_step(_STT, gen, mimi, block)
                    if piece:
                        text += piece
                        loop.call_soon_threadsafe(out_q.put_nowait, ("partial", text))
            finally:
                loop.call_soon_threadsafe(out_q.put_nowait, ("final", text))

    worker_task = asyncio.create_task(asyncio.to_thread(worker))
    buf = np.zeros(0, np.float32)
    received = 0  # STT_SR samples fed, for a quick "did we actually get audio?" log
    try:
        async for msg in ws:
            data = json.loads(msg)
            t = data.get("type")
            if t == "start":
                buf = np.zeros(0, np.float32)
            elif t == "audio":
                pcm = np.frombuffer(base64.b64decode(data["data"]), "<i2").astype(np.float32) / 32768.0
                res = _resample(pcm, 16000, STT_SR)
                received += len(res)
                buf = np.concatenate([buf, res])
                while len(buf) >= BLOCK:
                    in_q.put(buf[:BLOCK].copy())
                    buf = buf[BLOCK:]
                while not out_q.empty():
                    kind, txt = out_q.get_nowait()
                    if kind == "partial":
                        await ws.send(json.dumps({"type": "partial", "text": txt.strip()}))
            elif t == "stop":
                log.info("STT stop: %.2fs of audio received", received / STT_SR)
                if len(buf) > 0:
                    in_q.put(np.pad(buf, (0, BLOCK - len(buf))))  # pad the trailing partial block
                for _ in range(FLUSH_BLOCKS):  # drain the lookahead delay so the last words emit
                    in_q.put(np.zeros(BLOCK, np.float32))
                break
    finally:
        in_q.put(None)  # always release the worker, even if the socket dropped

    await worker_task
    final_text = ""
    while not out_q.empty():
        kind, txt = out_q.get_nowait()
        final_text = txt
    log.info("STT final: %r", final_text.strip())
    await ws.send(json.dumps({"type": "final", "text": final_text.strip()}))


# --- TTS (Kokoro) ------------------------------------------------------------

def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        from mlx_audio.tts.utils import load_model

        log.info("loading Kokoro TTS (%s)...", KOKORO_MODEL)
        _kokoro = load_model(KOKORO_MODEL)
    return _kokoro


def _synth(text: str, voice: str | None, emit):
    """Synthesize text locally, emitting Kokoro's generated audio segments."""
    with _mlx_lock:
        model = _get_kokoro()
        audio_seconds = 0.0
        voice_id = voice or KOKORO_VOICE
        started = time.perf_counter()
        for result in model.generate(
            text=text, voice=voice_id, speed=KOKORO_SPEED, lang_code=voice_id[0],
        ):
            samples = np.array(result.audio, dtype=np.float32, copy=True)
            sample_rate = int(getattr(result, "sample_rate", 24000))
            audio_seconds += samples.size / sample_rate
            emit(samples, sample_rate)
        elapsed = time.perf_counter() - started
        if audio_seconds:
            log.info(
                "TTS generated %.2fs audio in %.2fs (%.2fx realtime)",
                audio_seconds, elapsed, elapsed / audio_seconds,
            )


def _take_tts_chunks(text: str) -> tuple[list[str], str]:
    """Take completed sentences or longer clauses from buffered text."""
    chunks = []
    while text:
        sentence_end = TTS_SENTENCE_END.search(text)
        clause_end = TTS_CLAUSE_END.search(text)
        boundaries = []
        if sentence_end and sentence_end.end() <= TTS_MAX_BUFFER_CHARS:
            boundaries.append(sentence_end.end())
        if (
            clause_end
            and TTS_MIN_CLAUSE_CHARS <= clause_end.end() <= TTS_MAX_BUFFER_CHARS
        ):
            boundaries.append(clause_end.end())

        if boundaries:
            cut = min(boundaries)
        elif len(text) > TTS_MAX_BUFFER_CHARS:
            cut = text.rfind(" ", 0, TTS_MAX_BUFFER_CHARS + 1)
            if cut <= 0:
                cut = text.find(" ", TTS_MAX_BUFFER_CHARS)
                if cut < 0:
                    break
            cut += 1
        else:
            break
        chunks.append(text[:cut])
        text = text[cut:]
    return chunks, text


async def _tts_session(ws):
    buf: list[str] = []
    voice = None
    loop = asyncio.get_running_loop()

    async def synth_text(text: str):
        text = text.strip()
        if not text:
            return
        q: asyncio.Queue = asyncio.Queue()
        emit = lambda samples, sr: loop.call_soon_threadsafe(q.put_nowait, (samples, sr))

        def produce():
            try:
                _synth(text, voice, emit)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        worker = asyncio.create_task(asyncio.to_thread(produce))
        while (item := await q.get()) is not None:
            await ws.send(_audio_frame(*item))
        await worker

    async def synth_buffered():
        text = "".join(buf)
        buf.clear()
        await synth_text(text)

    async for msg in ws:
        data = json.loads(msg)
        t = data.get("type")
        if t == "start":
            voice = data.get("voice")
            buf.clear()
        elif t == "text":
            buf.append(data.get("text", ""))
            chunks, remainder = _take_tts_chunks("".join(buf))
            buf[:] = [remainder]
            for chunk in chunks:
                await synth_text(chunk)
        elif t == "flush":
            await synth_buffered()
        elif t == "stop":
            await synth_buffered()
            await ws.send(json.dumps({"type": "final"}))
            return


# --- wiring ------------------------------------------------------------------

async def router(ws):
    path = ws.request.path
    try:
        if path == "/stt":
            await _stt_session(ws)
        elif path == "/tts":
            await _tts_session(ws)
        else:
            await ws.close(code=1008, reason="unknown path")
    except Exception as e:
        log.exception("session failed on %s", path)
        try:
            await ws.send(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass


async def main():
    global _STT
    log.info("loading Kyutai STT (%s)...", STT_REPO)
    _STT = _load_stt()
    log.info(
        "STT warm. Kokoro TTS loads locally on first use (speed=%.2fx).",
        KOKORO_SPEED,
    )
    # max_size=None: base64 PCM frames exceed the 1 MB default; localhost, trusted.
    async with websockets.serve(router, HOST, PORT, max_size=None):
        log.info("listening on ws://%s:%d  (/stt, /tts)", HOST, PORT)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
