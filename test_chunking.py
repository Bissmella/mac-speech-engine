"""Checks the TTS sentence/clause chunker (pure logic, no models loaded).

Run on the Mac where MLX is installed: `python test_chunking.py`.
Importing `server` pulls MLX but does not load any model (main() is guarded).
"""

from server import _take_tts_chunks, TTS_MAX_BUFFER_CHARS


def test_splits_each_sentence():
    assert _take_tts_chunks("Hello there. How are you? ") == (
        ["Hello there. ", "How are you? "], "")


def test_buffers_until_a_boundary():
    assert _take_tts_chunks("no boundary yet") == ([], "no boundary yet")


def test_short_clause_keeps_buffering():
    # a comma before the min-clause length must not emit a tiny fragment
    assert _take_tts_chunks("Tiny, bit") == ([], "Tiny, bit")


def test_force_splits_long_unpunctuated_text():
    chunks, remainder = _take_tts_chunks("word " * 50)  # 250 chars, no punctuation
    assert chunks, "long text must be cut, not held forever"
    assert all(len(c) <= TTS_MAX_BUFFER_CHARS + 1 for c in chunks), chunks
    assert remainder, "the tail stays buffered for more text"


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print("ok", _name)
    print("all chunker checks passed")
