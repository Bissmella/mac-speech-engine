# mac-speech-engine

Local STT + TTS WebSocket server for Apple Silicon (MLX). Kyutai STT for
streaming speech-to-text, Kyutai TTS (Kokoro as a fallback) for text-to-speech.
Localhost only, no auth. It's the server half of "Code Companion"; the VS Code
extension is the client.

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
python server.py        # loads STT + Kyutai TTS, then listens on ws://127.0.0.1:8765
```

Endpoints: `ws://127.0.0.1:8765/stt` and `/tts`. Kokoro loads lazily on first use.

## Acceptance check (no VS Code)

With the server running, in another terminal:

```bash
python test_client.py                      # TTS a sentence -> tts_out.wav, then STT it back
python test_client.py --tts-engine kokoro  # same, using the Kokoro fallback voice
python test_client.py --wav recording.wav  # STT your own 16 kHz mono WAV
```

It writes `tts_out.wav` (playable) and prints the live partials then the final transcript.

## Models / voices

| | Model | Notes |
|---|---|---|
| STT | `kyutai/stt-1b-en_fr-mlx` | 1B, ~0.5s delay, English + French, streaming. Input 16 kHz (resampled to 24 kHz internally) |
| TTS (default) | Kyutai DSM TTS | 24 kHz mono, streaming, default voice `expresso/ex03-...` |
| TTS (fallback) | `mlx-community/Kokoro-82M-bf16` | 24 kHz mono, default voice `af_heart` |

The TTS `start` frame picks the engine: `{"type":"start","engine":"kyutai"}` or
`"kokoro"`, with an optional `voice`. Default is `kyutai`. Kokoro is there in case
Kyutai's streaming TTS prosody disappoints ([issue #170](https://github.com/kyutai-labs/delayed-streams-modeling/issues/170)).
Swap the STT model (e.g. to `kyutai/stt-2.6b-en-mlx`, English-only but higher
latency) by editing the constants at the top of `server.py`.

## Protocol

Every frame is a JSON text frame. Audio is base64 raw PCM, 16-bit signed
little-endian, mono.

**STT** `/stt` — client sends `start` (`sample_rate` 16000, optional
`language`), repeated `audio` (`data`: base64 PCM), then `stop`. Server streams
zero or more `partial` (`text`) as you speak, then one `final` (`text`), or `error`.

**TTS** `/tts` — client sends `start` (`engine`, `voice`), repeated `text`, then
`flush` (synthesize now, keep session) or `stop` (synthesize remainder, then end).
Server streams ordered `audio` frames (`data`, `sample_rate`, `format`), ends
with `final`, or `error`.

## Status

The STT and TTS paths are ports of Kyutai's MLX reference scripts
(`delayed-streams-modeling`), pinned to `moshi_mlx==0.2.12`. They parse and
mirror the upstream API, but were authored on Windows and not run on MLX, so
confirm the `moshi_mlx` version and run the acceptance check on your Mac. Kokoro
is the proven fallback if the Kyutai path needs a version fix.

## Out of scope

Auth, multi-user, reconnection, non-localhost networking.
