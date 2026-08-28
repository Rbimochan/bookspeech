from pathlib import Path

import numpy as np
import pytest

from app.audio_io import read_wav, write_wav
from app.tts.base import SynthesisResult, TTSEngine, VoiceInfo
from app.tts.xtts_engine import XTTSEngine


class FakeEngine(TTSEngine):
    """Exercises the TTSEngine contract without needing real model weights."""

    def __init__(self, rms: float = 0.5):
        self._rms = rms

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> SynthesisResult:
        n = max(len(text), 1) * 100
        audio = np.full(n, self._rms, dtype=np.float32)
        return SynthesisResult(audio=audio, sample_rate=24000)

    def list_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id="fake_voice", name="Fake Voice", language="en-us", gender="female")]


def test_engine_implements_interface():
    engine = FakeEngine()
    result = engine.synthesize("hello world", voice="fake_voice")
    assert isinstance(result.audio, np.ndarray)
    assert result.sample_rate == 24000


def test_list_voices_returns_voice_info():
    engine = FakeEngine()
    voices = engine.list_voices()
    assert len(voices) == 1
    assert voices[0].id == "fake_voice"


def test_xtts_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        XTTSEngine()


def test_write_and_read_wav_roundtrip(tmp_path):
    audio = np.sin(np.linspace(0, 10, 2400)).astype(np.float32) * 0.5
    path = tmp_path / "chunk.wav"
    write_wav(path, audio, sample_rate=24000)

    assert path.exists()
    read_audio, sample_rate = read_wav(path)
    assert sample_rate == 24000
    assert len(read_audio) == len(audio)
    # int16 quantization introduces small error; allow a loose tolerance.
    assert np.allclose(read_audio, audio, atol=1e-3)


def test_write_wav_creates_parent_dirs(tmp_path):
    audio = np.zeros(100, dtype=np.float32)
    path = tmp_path / "nested" / "chunks" / "0.wav"
    write_wav(path, audio, sample_rate=24000)
    assert path.exists()


_MODEL_DIR = Path(__file__).parent.parent / "backend" / "models"


@pytest.mark.skipif(
    not (_MODEL_DIR / "kokoro-v1.0.onnx").exists(),
    reason="Kokoro model weights not downloaded; see README.md",
)
def test_kokoro_engine_real_synthesis(tmp_path):
    from app.tts.kokoro_engine import KokoroEngine

    engine = KokoroEngine(_MODEL_DIR / "kokoro-v1.0.onnx", _MODEL_DIR / "voices-v1.0.bin")
    voices = engine.list_voices()
    assert len(voices) > 0

    out_path = tmp_path / "test.wav"
    engine.synthesize_chunk_to_file("Hello, this is a test.", voices[0].id, 1.0, out_path)
    audio, sr = read_wav(out_path)
    assert len(audio) > 0
    assert sr > 0


# ---------------------------------------------------------------------------
# Punctuation pacing: kokoro-onnx only inserts its own sentence/clause pauses
# when it's forced to split text into multiple batches (~510 phonemes), so a
# normal-sized chunk gets zero pause on internal periods/commas by default.
# KokoroEngine.synthesize works around this by always segmenting on sentence/
# clause marks and inserting real silence itself.
# ---------------------------------------------------------------------------

def test_split_into_speech_segments_splits_on_sentence_and_clause_marks():
    from app.tts.kokoro_engine import _split_into_speech_segments

    text = "First sentence. Second, with a clause. Third — with a dash."
    segments = _split_into_speech_segments(text)
    assert segments == [
        "First sentence.",
        "Second,",
        "with a clause.",
        "Third —",
        "with a dash.",
    ]


def test_split_into_speech_segments_single_sentence_stays_one_segment():
    from app.tts.kokoro_engine import _split_into_speech_segments

    assert _split_into_speech_segments("Just one sentence with no breaks") == [
        "Just one sentence with no breaks"
    ]


# ---------------------------------------------------------------------------
# Regression: a text fragment with no letters/digits at all (a stray "]" left
# over from a malformed footnote marker, "..." after cleaning, etc.) crashed
# the whole job — kokoro-onnx raises ValueError("... produced no phonemes")
# instead of returning silence for punctuation-only input.
# ---------------------------------------------------------------------------

def test_split_into_speech_segments_drops_punctuation_only_fragments():
    from app.tts.kokoro_engine import _split_into_speech_segments

    text = "Real sentence here. ] Another real sentence."
    segments = _split_into_speech_segments(text)
    assert "]" not in segments
    assert all(any(c.isalnum() for c in s) for s in segments)


def test_split_into_speech_segments_all_punctuation_returns_empty():
    from app.tts.kokoro_engine import _split_into_speech_segments

    assert _split_into_speech_segments("] ... !!") == []


def test_pause_after_segment_uses_sentence_pause_for_terminal_punctuation():
    from app.tts.kokoro_engine import _pause_after_segment

    assert _pause_after_segment("Done.", sentence_pause=0.4, clause_pause=0.18) == 0.4
    assert _pause_after_segment("Done!", sentence_pause=0.4, clause_pause=0.18) == 0.4
    assert _pause_after_segment("Done?", sentence_pause=0.4, clause_pause=0.18) == 0.4


def test_pause_after_segment_uses_clause_pause_for_commas_and_dashes():
    from app.tts.kokoro_engine import _pause_after_segment

    assert _pause_after_segment("wait,", sentence_pause=0.4, clause_pause=0.18) == 0.18
    assert _pause_after_segment("wait;", sentence_pause=0.4, clause_pause=0.18) == 0.18
    assert _pause_after_segment("wait —", sentence_pause=0.4, clause_pause=0.18) == 0.18


def test_pause_after_segment_no_pause_without_punctuation():
    from app.tts.kokoro_engine import _pause_after_segment

    assert _pause_after_segment("no punctuation here", sentence_pause=0.4, clause_pause=0.18) == 0.0


@pytest.mark.skipif(
    not (_MODEL_DIR / "kokoro-v1.0.onnx").exists(),
    reason="Kokoro model weights not downloaded; see README.md",
)
def test_kokoro_engine_inserts_real_pauses_at_punctuation():
    """A multi-sentence chunk synthesized with pacing must be measurably
    longer than the same text with pacing disabled (zero pause) — otherwise
    the punctuation-driven silence isn't actually being inserted."""
    from app.tts.kokoro_engine import KokoroEngine

    engine = KokoroEngine(_MODEL_DIR / "kokoro-v1.0.onnx", _MODEL_DIR / "voices-v1.0.bin")
    voice = engine.list_voices()[0].id
    text = "First sentence here. Second sentence, with a clause, follows it."

    no_pause = engine.synthesize(text, voice, speed=1.0, sentence_pause=0.0, clause_pause=0.0)
    with_pause = engine.synthesize(text, voice, speed=1.0, sentence_pause=0.4, clause_pause=0.18)

    no_pause_duration = len(no_pause.audio) / no_pause.sample_rate
    with_pause_duration = len(with_pause.audio) / with_pause.sample_rate
    assert with_pause_duration > no_pause_duration + 0.5  # at least most of the 0.76s of pauses


@pytest.mark.skipif(
    not (_MODEL_DIR / "kokoro-v1.0.onnx").exists(),
    reason="Kokoro model weights not downloaded; see README.md",
)
def test_kokoro_engine_survives_punctuation_only_chunk():
    """Regression: a chunk that's just a stray bracket (no letters/digits at
    all) crashed the whole job with ValueError("... produced no phonemes").
    It must produce silence instead."""
    from app.tts.kokoro_engine import KokoroEngine

    engine = KokoroEngine(_MODEL_DIR / "kokoro-v1.0.onnx", _MODEL_DIR / "voices-v1.0.bin")
    voice = engine.list_voices()[0].id

    result = engine.synthesize("]", voice, speed=1.0)
    assert isinstance(result.audio, np.ndarray)  # didn't raise

    # And a real chunk that happens to contain one, mixed with real content.
    result2 = engine.synthesize("Real sentence here. ] Another real sentence.", voice, speed=1.0)
    assert len(result2.audio) > 0
