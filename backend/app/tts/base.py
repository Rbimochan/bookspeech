"""Abstract TTS engine interface. Every synthesis backend (Kokoro today,
XTTS-v2 later) implements this so the rest of the pipeline never depends on
a specific engine's API."""

from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel


class VoiceInfo(BaseModel):
    id: str
    name: str
    language: str | None = None
    gender: str | None = None


class SynthesisResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    audio: np.ndarray
    sample_rate: int


class TTSEngine(ABC):
    @abstractmethod
    def synthesize(self, text: str, voice: str, speed: float = 1.0) -> SynthesisResult:
        """Synthesize one chunk of text. Returns mono float32 PCM audio and its sample rate."""

    @abstractmethod
    def list_voices(self) -> list[VoiceInfo]:
        """List voices available for UI voice selection."""
