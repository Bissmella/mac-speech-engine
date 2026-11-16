"""Local STT+TTS WebSocket server for Apple Silicon (MLX).

/stt  Kyutai STT (streaming). Resamples the 16 kHz wire input to 24 kHz, feeds
      80 ms blocks, emits `partial` frames live and one `final` on stop.
/tts  proxies a streaming moshi-server (Rust/Metal) so audio starts before the full
      text arrives. Streams audio frames as produced.

Localhost only, one connection = one session. See README.

NOTE: STT is a structural port of kyutai-labs/delayed-streams-modeling MLX scripts
(moshi_mlx 0.2.12), verified against the scripts but not run on this (non-macOS)
machine. Kyutai TTS uses moshi-server's /api/tts_streaming protocol (see the unmute
repo). Confirm the moshi_mlx version and that moshi-server is reachable on first launch.
"""

import asyncio
import base64
import json
import logging
import os
import queue
import threading
import urllib.parse

import msgpack
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

STT_REPO = "kyutai/stt-2.6b-en-mlx"           # 2.6B English, ~2.5s delay (more accurate); 1b-en_fr is 0.5s

# TTS streams from a separate moshi-server (Rust/Metal). Run one and point KYUTAI_TTS_URL
# at it; "public_token" is moshi-server's default dev key.
KYUTAI_TTS_URL = os.environ.get("KYUTAI_TTS_URL", "ws://127.0.0.1:8089")
KYUTAI_TTS_API_KEY = os.environ.get("KYUTAI_TTS_API_KEY", "public_token")
TTS_STREAMING_PATH = "/api/tts_streaming"
TTS_CFG_ALPHA = 1.5
MOSHI_SR = 24000          # moshi-server emits 24 kHz float PCM

STT_SR = 24000          # Kyutai STT operates at 24 kHz
BLOCK = 1920            # 80 ms at 24 kHz, the frame size the model steps on
FLUSH_BLOCKS = 40       # ~3.2s of trailing silence to drain the 2.6b model's ~2.5s lookahead; tune per model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("speech")

# MLX isn't safe under concurrent model calls; serialize all inference across threads.
# ponytail: one lock, held per STT utterance / per TTS synth. Fine for localhost half-duplex use.
_mlx_lock = threading.Lock()

_STT = None          # loaded at startup


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


# --- TTS ---------------------------------------------------------------------
# Kyutai TTS streams text->audio only through Kyutai's moshi-server (Rust/Metal); the
# in-process moshi_mlx TTS API is whole-script. So /tts just bridges a running
# moshi-server, translating our JSON <-> its msgpack.

async def _tts_proxy_kyutai(ws, voice: str | None):
    """Bridge one /tts session to a streaming moshi-server: our JSON <-> its msgpack.
    Text flows up as it arrives and audio streams back before the full text is known."""
    params = {"format": "PcmMessagePack", "cfg_alpha": TTS_CFG_ALPHA}
    if voice:
        params["voice"] = voice
    url = f"{KYUTAI_TTS_URL}{TTS_STREAMING_PATH}?{urllib.parse.urlencode(params)}"
    headers = {"kyutai-api-key": KYUTAI_TTS_API_KEY}

    async with websockets.connect(url, additional_headers=headers, max_size=None) as up:
        async for raw in up:  # handshake: server sends Ready (and may replay stale frames) first
            m = msgpack.unpackb(raw)
            if m.get("type") == "Ready":
                break
            if m.get("type") == "Error":
                raise RuntimeError(f"moshi-server TTS error: {m.get('message')}")

        async def uplink():
            # Forward words as they arrive; Eos ends the turn. If our client closes first
            # (barge-in), the finally still sends Eos so the server's turn ends.
            try:
                async for raw in ws:
                    data = json.loads(raw)
                    t = data.get("type")
                    if t == "text":
                        await up.send(msgpack.packb({"type": "Text", "text": data.get("text", "")}))
                    elif t == "stop":
                        break
                    # "flush" is a no-op: the server streams continuously
            finally:
                try:
                    await up.send(msgpack.packb({"type": "Eos"}))
                except Exception:
                    pass

        up_task = asyncio.create_task(uplink())
        try:
            async for raw in up:  # audio streams back; the server closes once Eos is drained
                m = msgpack.unpackb(raw)
                if m.get("type") == "Audio":
                    await ws.send(_audio_frame(np.asarray(m["pcm"], dtype=np.float32), MOSHI_SR))
                elif m.get("type") == "Error":
                    await ws.send(json.dumps({"type": "error", "message": str(m.get("message"))}))
                    return
        finally:
            up_task.cancel()
        await ws.send(json.dumps({"type": "final"}))


async def _tts_session(ws):
    # The first frame is 'start'; everything after streams through the moshi-server proxy.
    async for msg in ws:
        data = json.loads(msg)
        if data.get("type") == "start":
            await _tts_proxy_kyutai(ws, data.get("voice"))
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
    log.info("STT warm. TTS streams from moshi-server at %s.", KYUTAI_TTS_URL)
    # max_size=None: base64 PCM frames exceed the 1 MB default; localhost, trusted.
    async with websockets.serve(router, HOST, PORT, max_size=None):
        log.info("listening on ws://%s:%d  (/stt, /tts)", HOST, PORT)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
