"""Manual QA checklist: synthesize the preview line for every bundled Kokoro
voice and flag any that sound broken (silent, near-silent, or errored) so a
human doesn't have to guess which ones need listening to.

Usage:
    cd backend
    uv run python ../scripts/preview_all_voices.py
    afplay /tmp/bookspeech_voice_qa/af_heart.wav   # spot-check any voice
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

import numpy as np  # noqa: E402

from app.routers.voices import DEFAULT_PREVIEW_TEXT  # noqa: E402
from app.tts.kokoro_engine import KokoroEngine  # noqa: E402

MODEL_DIR = Path(__file__).parent.parent / "backend" / "models"
OUT_DIR = Path("/tmp/bookspeech_voice_qa")
SILENCE_RMS_THRESHOLD = 0.005


def main() -> None:
    engine = KokoroEngine(MODEL_DIR / "kokoro-v1.0.onnx", MODEL_DIR / "voices-v1.0.bin")
    voices = engine.list_voices()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Previewing {len(voices)} voices with: {DEFAULT_PREVIEW_TEXT!r}\n")
    flagged = []
    for v in voices:
        out_path = OUT_DIR / f"{v.id}.wav"
        try:
            result = engine.synthesize(DEFAULT_PREVIEW_TEXT, voice=v.id, speed=1.0)
            rms = float(np.sqrt(np.mean(np.square(result.audio)))) if len(result.audio) else 0.0
            from app.audio_io import write_wav

            write_wav(out_path, result.audio, result.sample_rate)
            ok = rms >= SILENCE_RMS_THRESHOLD
            status = "OK  " if ok else "FLAG"
            print(f"[{status}] {v.id:16s} rms={rms:.4f}  -> {out_path}")
            if not ok:
                flagged.append(v.id)
        except Exception as e:
            print(f"[FAIL] {v.id:16s} error: {e}")
            flagged.append(v.id)

    print()
    if flagged:
        print(f"Flagged {len(flagged)} voice(s) for manual listening: {', '.join(flagged)}")
    else:
        print("All voices produced non-silent output. Spot-check a few by ear before shipping.")


if __name__ == "__main__":
    main()
