# mac-speech-engine

Local STT + TTS WebSocket server for Apple Silicon (MLX). Kyutai STT runs
in-process for streaming speech-to-text. For streaming text-to-speech it bridges
a separate Kyutai moshi-server. Localhost only, no auth. It's the server half of
"Code Companion"; the VS Code extension is the client.

## Requirements

- macOS, Apple Silicon (M-series). Fits an M4 Pro with room to spare.
- Python 3.12 (moshi_mlx targets 3.12)

## Install

```bash
pip install -r requirements.txt
```

Models download from Hugging Face on first run (~6 GB total, cached after).

## Run

```bash
python server.py        # loads STT, then listens on ws://127.0.0.1:8765
```

Endpoints: `ws://127.0.0.1:8765/stt` and `/tts`. STT runs here; `/tts` bridges a moshi-server.

### Streaming TTS needs a moshi-server

`/tts` does not synthesize in-process — it bridges a Kyutai **moshi-server** that
streams text in and audio out. Run one (see the `unmute` repo and kyutai-labs/moshi
for the Rust `moshi-server` and a TTS config) and set `KYUTAI_TTS_URL` (default
`ws://127.0.0.1:8089`). `/tts` is unavailable until that server is reachable.

## Acceptance check (no VS Code)

With the server running, in another terminal:

```bash
python test_client.py                      # TTS -> tts_out.wav, then STT it back (needs moshi-server)
python test_client.py --wav recording.wav  # STT your own 16 kHz mono WAV (no moshi-server needed)
```

It writes `tts_out.wav` (playable) and prints the live partials then the final transcript.

## Models / voices

| | Model | Notes |
|---|---|---|
| STT | `kyutai/stt-1b-en_fr-mlx` | 1B, English + French, ~0.5s delay, streaming, has a semantic VAD. Input 16 kHz (resampled to 24 kHz internally) |
| TTS | Kyutai DSM TTS via **moshi-server** | 24 kHz mono, true text streaming. Bridges a moshi-server at `KYUTAI_TTS_URL` (default `ws://127.0.0.1:8089`) |

The TTS `start` frame takes an optional `voice` (a moshi-server voice id; omit for
the server default). Swap the STT model (e.g. to `kyutai/stt-2.6b-en-mlx`,
English-only, more accurate but ~2.5s delay) by editing the constants at the top of
`server.py` — remember to retune `FLUSH_BLOCKS` to the model's delay.

## Protocol

Every frame is a JSON text frame. Audio is base64 raw PCM, 16-bit signed
little-endian, mono.

**STT** `/stt` — client sends `start` (`sample_rate` 16000, optional
`language`), repeated `audio` (`data`: base64 PCM), then `stop`. Server streams
zero or more `partial` (`text`) as you speak, then one `final` (`text`), or `error`.

**TTS** `/tts` — client sends `start` (optional `voice`), repeated `text`, then
`stop`. The server bridges a moshi-server that streams `audio` frames (`data`,
`sample_rate`, `format`) as text arrives, ending with `final`, or `error`.

## Status

STT is a port of Kyutai's MLX reference (`delayed-streams-modeling`,
`moshi_mlx==0.2.12`), mirroring the upstream API but authored on Windows and not
run on MLX — confirm the version and run the acceptance check on your Mac.
TTS depends on a reachable moshi-server (`/api/tts_streaming`, the same protocol
`unmute` uses); it is a thin bridge, verified against unmute's client but not run
here.

## Out of scope

Auth, multi-user, reconnection, non-localhost networking.
