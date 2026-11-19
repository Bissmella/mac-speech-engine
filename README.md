# mac-speech-engine

Local STT + TTS WebSocket server for Apple Silicon (MLX). Kyutai STT and Kokoro
TTS run in-process. Localhost only, no auth. It's the server half of "Code
Companion"; the VS Code extension is the client.

## Requirements

- macOS, Apple Silicon (M-series). Fits an M4 Pro with room to spare.
- Python 3.12 (moshi_mlx targets 3.12)

## Install

```bash
pip install -r requirements.txt
```

Models download from Hugging Face as needed and are cached locally.

## Run

```bash
python server.py        # loads STT, then listens on ws://127.0.0.1:8765
```

Endpoints: `ws://127.0.0.1:8765/stt` and `/tts`. TTS loads its model lazily on
the first `/tts` request; no separate `moshi-server` process is needed. It starts
synthesizing complete sentences and longer clauses as text arrives; unpunctuated
text is split at about 160 characters. Its speaking rate defaults to `1.25x`; set
`KOKORO_SPEED=1.50` before starting `server.py` to try `1.50x` instead.

## Acceptance check (no VS Code)

With the server running, in another terminal:

```bash
python test_client.py                      # TTS -> tts_out.wav, then STT it back
python test_client.py --wav recording.wav  # STT your own 16 kHz mono WAV
```

It writes `tts_out.wav` (playable) and prints the live partials then the final transcript.

## Models / voices

| | Model | Notes |
|---|---|---|
| STT | `kyutai/stt-1b-en_fr-mlx` | 1B, English + French, ~0.5s delay, streaming, has a semantic VAD. Input 16 kHz (resampled to 24 kHz internally) |
| TTS | Kokoro (`mlx-community/Kokoro-82M-bf16`) | Local MLX inference, 24 kHz mono, default speed `1.25x`. The default voice is `af_heart`. |

The TTS `start` frame takes an optional Kokoro voice id (e.g. `af_heart`; omit
for the default). Swap the STT model (e.g. to `kyutai/stt-2.6b-en-mlx`,
English-only, more accurate but ~2.5s delay) by editing the constants at the top of
`server.py` — remember to retune `FLUSH_BLOCKS` to the model's delay.

## Protocol

Every frame is a JSON text frame. Audio is base64 raw PCM, 16-bit signed
little-endian, mono.

**STT** `/stt` — client sends `start` (`sample_rate` 16000, optional
`language`), repeated `audio` (`data`: base64 PCM), then `stop`. Server streams
zero or more `partial` (`text`) as you speak, then one `final` (`text`), or `error`.

**TTS** `/tts` — client sends `start` (optional `voice`), repeated `text`, then
optional `flush` frames to synthesize buffered text or `stop` to synthesize any
remaining text and end the session. The server synthesizes complete sentences
and longer clauses as text arrives, streams `audio` frames (`data`, `sample_rate`,
`format`) as the local model generates them, then ends with `final` or `error`.

## Status

STT is a port of Kyutai's MLX reference (`delayed-streams-modeling`,
`moshi_mlx==0.2.12`). TTS uses MLX Audio's Kokoro model. Both need an Apple
Silicon Mac with MLX; model weights download from Hugging Face on demand.

## Out of scope

Auth, multi-user, reconnection, non-localhost networking.
