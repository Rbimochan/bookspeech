"""Kokoro-82M TTS engine, via the kokoro-onnx runtime."""

import logging
import re
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro
from kokoro_onnx.session import resolve_providers

from app.audio_io import write_wav
from app.tts.base import SynthesisResult, TTSEngine, VoiceInfo

logger = logging.getLogger(__name__)

# Below this RMS, output is treated as near-silent/garbage and retried.
_SILENCE_RMS_THRESHOLD = 0.001
_MAX_RETRIES = 2

# kokoro-onnx only inserts its own sentence/clause pauses at the batch splits
# it makes when text exceeds ~510 phonemes (see kokoro_onnx/chunker.py) — for
# a typical TTS chunk (well under that limit) it's synthesized as one single
# pass with NO pause on internal periods/commas at all, so the narration
# reads flat and fast no matter how the text is punctuated. We drive the same
# sentence/clause pausing ourselves, unconditionally, by splitting each chunk
# into segments at those marks and stitching them with explicit silence.
_SENTENCE_MARKS = ".!?…"
_CLAUSE_MARKS = ",;:—"
_SEGMENT_SPLIT_RE = re.compile(rf"(?<=[{re.escape(_SENTENCE_MARKS + _CLAUSE_MARKS)}])\s+")

DEFAULT_SENTENCE_PAUSE_SEC = 0.4
DEFAULT_CLAUSE_PAUSE_SEC = 0.18

# A fragment with no letters or digits (stray "]" from a malformed footnote
# marker, a lone "..." left after cleaning, etc.) has nothing for the model
# to actually say — Kokoro raises ValueError("... produced no phonemes") on
# these rather than returning silence, which crashed the whole job on a
# single bad text fragment. Filter them out before they ever reach the engine.
_HAS_SPEAKABLE_CONTENT_RE = re.compile(r"[a-zA-Z0-9]")


def _split_into_speech_segments(text: str) -> list[str]:
    return [
        s
        for s in _SEGMENT_SPLIT_RE.split(text)
        if s.strip() and _HAS_SPEAKABLE_CONTENT_RE.search(s)
    ]


def _pause_after_segment(segment: str, sentence_pause: float, clause_pause: float) -> float:
    mark = segment.rstrip()[-1:]
    if mark in _SENTENCE_MARKS:
        return sentence_pause
    if mark in _CLAUSE_MARKS:
        return clause_pause
    return 0.0

# Kokoro voice ids encode language+gender as a two-letter prefix, e.g.
# "af_heart" = American Female "Heart". Used only to populate VoiceInfo
# metadata for the UI; the id itself is passed straight through to Kokoro.
_LANG_PREFIX = {"a": "en-us", "b": "en-gb", "j": "ja", "z": "zh", "e": "es", "f": "fr", "h": "hi", "i": "it", "p": "pt-br"}
_GENDER_PREFIX = {"f": "female", "m": "male"}


def _voice_info_from_id(voice_id: str) -> VoiceInfo:
    name = voice_id.split("_", 1)[-1].replace("_", " ").title()
    lang = gender = None
    if len(voice_id) >= 2 and voice_id[1] in _GENDER_PREFIX:
        lang = _LANG_PREFIX.get(voice_id[0])
        gender = _GENDER_PREFIX.get(voice_id[1])
    return VoiceInfo(id=voice_id, name=name, language=lang, gender=gender)


class KokoroEngine(TTSEngine):
    def __init__(self, model_path: str | Path, voices_path: str | Path):
        model_path, voices_path = Path(model_path), Path(voices_path)
        if not model_path.exists() or not voices_path.exists():
            raise FileNotFoundError(
                f"Kokoro model weights not found at {model_path} / {voices_path}. "
                "Download them per README.md before starting the backend."
            )
        # kokoro-onnx auto-detects GPU vs CPU providers on load (respects the
        # ONNX_PROVIDER env var and picks up onnxruntime-gpu if installed).
        logger.info("Loading Kokoro model, providers=%s", resolve_providers())
        self._kokoro = Kokoro(str(model_path), str(voices_path))

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        sentence_pause: float = DEFAULT_SENTENCE_PAUSE_SEC,
        clause_pause: float = DEFAULT_CLAUSE_PAUSE_SEC,
    ) -> SynthesisResult:
        segments = _split_into_speech_segments(text)
        if not segments:
            # The whole chunk was punctuation/whitespace with nothing to
            # speak (e.g. a stray "]" left over from a malformed footnote
            # marker) — silence is the correct output, not a crash.
            logger.warning("Chunk had no speakable content, emitting silence: %r", text[:80])
            return SynthesisResult(audio=np.zeros(0, dtype=np.float32), sample_rate=24000)
        if len(segments) == 1:
            return self._synthesize_raw(segments[0], voice, speed)

        audio_parts: list[np.ndarray] = []
        sample_rate = None
        for i, segment in enumerate(segments):
            result = self._synthesize_raw(segment, voice, speed)
            sample_rate = result.sample_rate
            audio_parts.append(result.audio)
            if i < len(segments) - 1:
                pause_sec = _pause_after_segment(segment, sentence_pause, clause_pause)
                if pause_sec > 0:
                    audio_parts.append(np.zeros(int(pause_sec * sample_rate), dtype=np.float32))

        return SynthesisResult(audio=np.concatenate(audio_parts), sample_rate=sample_rate)

    def _synthesize_raw(self, text: str, voice: str, speed: float) -> SynthesisResult:
        last_audio, last_sr = None, None
        for attempt in range(1, _MAX_RETRIES + 2):
            try:
                audio, sample_rate = self._kokoro.create(text, voice=voice, speed=speed, lang="en-us")
            except ValueError as e:
                # Defensive fallback: _split_into_speech_segments filters out
                # unspeakable fragments, but this guards any pattern that
                # slips through (e.g. punctuation kokoro's own phonemizer
                # treats as empty) so one bad fragment can't kill the job.
                if "produced no phonemes" in str(e):
                    logger.warning("No phonemes for %r, emitting silence instead of failing the job", text[:80])
                    return SynthesisResult(audio=np.zeros(0, dtype=np.float32), sample_rate=24000)
                raise
            rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
            if rms >= _SILENCE_RMS_THRESHOLD:
                return SynthesisResult(audio=audio, sample_rate=sample_rate)
            logger.warning("Near-silent synthesis output (rms=%.5f) on attempt %d/%d", rms, attempt, _MAX_RETRIES + 1)
            last_audio, last_sr = audio, sample_rate
        # Exhausted retries — return the last (likely near-silent) result rather
        # than raising, so a single bad chunk doesn't abort the whole book;
        # the caller/job log records the warning trail above.
        return SynthesisResult(audio=last_audio, sample_rate=last_sr)

    def list_voices(self) -> list[VoiceInfo]:
        return [_voice_info_from_id(v) for v in self._kokoro.get_voices()]

    def synthesize_chunk_to_file(self, text: str, voice: str, speed: float, out_path: str | Path) -> Path:
        result = self.synthesize(text, voice, speed)
        write_wav(out_path, result.audio, result.sample_rate)
        return Path(out_path)
