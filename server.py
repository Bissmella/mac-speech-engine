"""Local STT+TTS WebSocket server for Apple Silicon (MLX).

/stt  Kyutai STT (streaming). Resamples the 16 kHz wire input to 24 kHz, feeds
      80 ms blocks, emits `partial` frames live and one `final` on stop.
/tts  Kyutai TTS by default, Kokoro as a fallback engine. Buffers text,
      synthesizes on flush/stop, streams audio frames as produced.

Localhost only, one connection = one session. See README.

NOTE: ported from kyutai-labs/delayed-streams-modeling MLX scripts and pinned to
moshi_mlx 0.2.12. The STT/TTS paths use that package's lower-level streaming API,
so they are structural ports verified against the scripts, not run on this
(non-macOS) machine. Expect to confirm the moshi_mlx version on first launch.
"""

import asyncio
import base64
import json
import logging
import queue
import threading

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

STT_REPO = "kyutai/stt-1b-en_fr-mlx"          # 1B, 0.5s delay (low latency); 2.6b-en is 2.5s
DEFAULT_TTS_ENGINE = "kyutai"                 # "kyutai" or "kokoro"
KYUTAI_TTS_VOICE = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
KOKORO_MODEL = "mlx-community/Kokoro-82M-bf16"
KOKORO_VOICE = "af_heart"

STT_SR = 24000          # Kyutai STT operates at 24 kHz
BLOCK = 1920            # 80 ms at 24 kHz, the frame size the model steps on
FLUSH_BLOCKS = 16       # ~1.3s of trailing silence to drain the model's ~0.5s lookahead; tune per model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("speech")

# MLX isn't safe under concurrent model calls; serialize all inference across threads.
# ponytail: one lock, held per STT utterance / per TTS synth. Fine for localhost half-duplex use.
_mlx_lock = threading.Lock()

_STT = None          # loaded at startup
_kyutai_tts = None   # lazy
_kokoro = None       # lazy


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
    try:
        async for msg in ws:
            data = json.loads(msg)
            t = data.get("type")
            if t == "start":
                buf = np.zeros(0, np.float32)
            elif t == "audio":
                pcm = np.frombuffer(base64.b64decode(data["data"]), "<i2").astype(np.float32) / 32768.0
                buf = np.concatenate([buf, _resample(pcm, 16000, STT_SR)])
                while len(buf) >= BLOCK:
                    in_q.put(buf[:BLOCK].copy())
                    buf = buf[BLOCK:]
                while not out_q.empty():
                    kind, txt = out_q.get_nowait()
                    if kind == "partial":
                        await ws.send(json.dumps({"type": "partial", "text": txt.strip()}))
            elif t == "stop":
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
    await ws.send(json.dumps({"type": "final", "text": final_text.strip()}))


# --- TTS (Kyutai + Kokoro) ---------------------------------------------------

def _load_kyutai_tts() -> dict:
    from moshi_mlx.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel
    from moshi_mlx.utils.loaders import hf_get

    raw = hf_get("config.json", DEFAULT_DSM_TTS_REPO)
    with open(hf_get(raw)) as f:
        raw = json.load(f)
    mimi_weights = hf_get(raw["mimi_name"], DEFAULT_DSM_TTS_REPO)
    moshi_weights = hf_get(raw.get("moshi_name", "model.safetensors"), DEFAULT_DSM_TTS_REPO)
    tokenizer = hf_get(raw["tokenizer_name"], DEFAULT_DSM_TTS_REPO)

    lm_config = models.LmConfig.from_config_dict(raw)
    lm_config.transformer.max_seq_len = lm_config.transformer.context  # moshi_mlx<=0.3.0 ring-cache workaround
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)
    model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

    text_tok = sentencepiece.SentencePieceProcessor(str(tokenizer))
    audio_tok = models.mimi.Mimi(models.mimi_202407(lm_config.generated_codebooks))
    audio_tok.load_pytorch_weights(str(mimi_weights), strict=True)

    tts = TTSModel(model, audio_tok, text_tok, voice_repo=DEFAULT_DSM_TTS_VOICE_REPO,
                   temp=0.6, cfg_coef=1, max_padding=8, initial_padding=2,
                   final_padding=2, padding_bonus=0, raw_config=raw)
    if tts.valid_cfg_conditionings:
        cfg_cond = tts.cfg_coef
        tts.cfg_coef = 1.0
        no_text = no_prefix = False
    else:
        cfg_cond = None
        no_text = no_prefix = True
    return {"tts": tts, "cfg_cond": cfg_cond, "no_text": no_text, "no_prefix": no_prefix}


def _get_kyutai_tts() -> dict:
    global _kyutai_tts
    if _kyutai_tts is None:
        log.info("loading Kyutai TTS...")
        _kyutai_tts = _load_kyutai_tts()
    return _kyutai_tts


def _get_kokoro():
    global _kokoro
    if _kokoro is None:
        log.info("loading Kokoro TTS...")
        from mlx_audio.tts.utils import load_model
        _kokoro = load_model(KOKORO_MODEL)
    return _kokoro


def _synth(engine: str, text: str, voice: str | None, emit):
    """Synthesize `text`, calling emit(float32_samples, sample_rate) per produced chunk."""
    with _mlx_lock:
        if engine == "kokoro":
            v = voice or KOKORO_VOICE
            for r in _get_kokoro().generate(text=text, voice=v, speed=1.0, lang_code=v[0]):
                emit(np.asarray(r.audio, dtype=np.float32), int(getattr(r, "sample_rate", 24000)))
            return

        k = _get_kyutai_tts()
        tts = k["tts"]
        entries = [tts.prepare_script([text])]
        voices = [tts.get_voice_path(voice or KYUTAI_TTS_VOICE)] if tts.multi_speaker else []
        attrs = [tts.make_condition_attributes(voices, k["cfg_cond"])]
        sr = int(tts.mimi.sample_rate)

        def on_frame(frame):
            if (frame == -1).any():
                return
            pcm = tts.mimi.decode_step(frame[:, :, None])
            emit(np.array(mx.clip(pcm[0, 0], -1, 1), dtype=np.float32), sr)

        tts.generate(entries, attrs, cfg_is_no_prefix=k["no_prefix"],
                     cfg_is_no_text=k["no_text"], on_frame=on_frame)


async def _tts_session(ws):
    buf: list[str] = []
    voice = None
    engine = DEFAULT_TTS_ENGINE
    loop = asyncio.get_running_loop()

    async def synth_buffered():
        text = "".join(buf).strip()
        buf.clear()
        if not text:
            return
        q: asyncio.Queue = asyncio.Queue()
        emit = lambda s, sr: loop.call_soon_threadsafe(q.put_nowait, (s, sr))

        def produce():
            try:
                _synth(engine, text, voice, emit)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        worker = asyncio.create_task(asyncio.to_thread(produce))
        while (item := await q.get()) is not None:
            await ws.send(_audio_frame(*item))
        await worker  # re-raise any synthesis error to the router

    async for msg in ws:
        data = json.loads(msg)
        t = data.get("type")
        if t == "start":
            engine = (data.get("engine") or DEFAULT_TTS_ENGINE).lower()
            voice = data.get("voice")  # None -> engine default
            buf.clear()
        elif t == "text":
            buf.append(data["text"])
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
    _get_kyutai_tts()  # warm the default TTS engine; Kokoro stays lazy (fallback only)
    log.info("models warm.")
    # max_size=None: base64 PCM frames exceed the 1 MB default; localhost, trusted.
    async with websockets.serve(router, HOST, PORT, max_size=None):
        log.info("listening on ws://%s:%d  (/stt, /tts)", HOST, PORT)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
