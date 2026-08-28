"""XTTS-v2 engine — not implemented yet.

Exists to prove TTSEngine is pluggable from day one: swapping the default
engine from Kokoro to XTTS should only require implementing this class and
changing config, with no changes to the pipeline/chunker/API layers.
"""

from app.tts.base import SynthesisResult, TTSEngine, VoiceInfo


class XTTSEngine(TTSEngine):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("XTTSEngine is not implemented yet; use KokoroEngine.")

    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> SynthesisResult:
        raise NotImplementedError

    def list_voices(self) -> list[VoiceInfo]:
        raise NotImplementedError
