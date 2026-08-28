"""Small helpers for reading/writing PCM wav files without extra deps."""

import wave
from pathlib import Path

import numpy as np


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767).astype(np.int16)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm16.tobytes())


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        sample_rate = f.getframerate()
        n_frames = f.getnframes()
        raw = f.readframes(n_frames)
    pcm16 = np.frombuffer(raw, dtype=np.int16)
    audio = pcm16.astype(np.float32) / 32767.0
    return audio, sample_rate
