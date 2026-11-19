"""Acceptance check: round-trips both endpoints with no VS Code.

TTS: sends a sentence, collects audio frames, writes tts_out.wav.
STT: streams 16 kHz PCM to /stt, prints the final transcript.
     By default it resamples the TTS output and feeds it back (self-contained,
     so TTS->STT is one chained proof). Pass --wav a.wav (16 kHz mono s16le)
     to transcribe a real recording instead.

    python test_client.py            # chained TTS -> STT
    python test_client.py --wav x.wav
"""

import argparse
import asyncio
import base64
import json
import wave

import numpy as np
import websockets

URL = "ws://127.0.0.1:8765"
SENTENCE = "The quick brown fox jumps over the lazy dog."


async def tts(text: str) -> tuple[bytes, int]:
    """Return (concatenated s16le PCM, sample rate) from the local TTS endpoint."""
    chunks, sr = [], 24000
    async with websockets.connect(URL + "/tts", max_size=None) as ws:
        await ws.send(json.dumps({"type": "start"}))
        await ws.send(json.dumps({"type": "text", "text": text}))
        await ws.send(json.dumps({"type": "stop"}))
        async for msg in ws:
            m = json.loads(msg)
            if m["type"] == "audio":
                chunks.append(base64.b64decode(m["data"]))
                sr = m["sample_rate"]
            elif m["type"] == "final":
                break
            elif m["type"] == "error":
                raise RuntimeError(m["message"])
    return b"".join(chunks), sr


async def stt(pcm16k: bytes, chunk=16000) -> str:
    """Stream 16 kHz mono s16le PCM, return the final transcript."""
    async with websockets.connect(URL + "/stt", max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "sample_rate": 16000,
                                  "format": "pcm_s16le", "language": "en"}))
        for i in range(0, len(pcm16k), chunk):
            frame = base64.b64encode(pcm16k[i:i + chunk]).decode("ascii")
            await ws.send(json.dumps({"type": "audio", "data": frame}))
        await ws.send(json.dumps({"type": "stop"}))
        async for msg in ws:
            m = json.loads(msg)
            if m["type"] == "final":
                return m["text"]
            if m["type"] == "error":
                raise RuntimeError(m["message"])
    return ""


def write_wav(path: str, pcm: bytes, sr: int):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)


def resample_s16le(pcm: bytes, src_sr: int, dst_sr: int) -> bytes:
    """Linear resample; good enough to feed the STT model for the acceptance check."""
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    n = int(len(x) * dst_sr / src_sr)
    y = np.interp(np.linspace(0, len(x), n, endpoint=False), np.arange(len(x)), x)
    return y.astype("<i2").tobytes()


def read_wav_16k(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, \
            "expected 16 kHz mono 16-bit WAV"
        return w.readframes(w.getnframes())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", help="16 kHz mono s16le WAV for STT (default: reuse TTS output)")
    args = ap.parse_args()

    print("TTS: synthesizing locally...")
    pcm, sr = await tts(SENTENCE)
    write_wav("tts_out.wav", pcm, sr)
    print(f"TTS: wrote tts_out.wav ({len(pcm)//2} samples @ {sr} Hz)")

    pcm16k = read_wav_16k(args.wav) if args.wav else resample_s16le(pcm, sr, 16000)
    print("STT: transcribing...")
    print("STT final:", repr(await stt(pcm16k)))


if __name__ == "__main__":
    asyncio.run(main())
