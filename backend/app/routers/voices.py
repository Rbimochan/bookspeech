import hashlib

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.config import settings

router = APIRouter()

# A short, fixed sentence used for every voice preview unless the caller
# supplies their own (e.g. their book's opening line) via ?text=.
DEFAULT_PREVIEW_TEXT = "The quick brown fox jumps over the lazy dog near the riverbank."


@router.get("/voices")
async def list_voices(request: Request):
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not loaded (Kokoro model weights missing)")
    return [v.model_dump() for v in engine.list_voices()]


# Implemented as GET (not the POST /voices/{id}/preview in the original plan
# doc) because the frontend plays it straight from an <audio src="...">,
# which can only issue GET requests.
@router.get("/voices/{voice_id}/preview")
async def preview_voice(voice_id: str, request: Request, text: str = "", speed: float = 1.0):
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not loaded (Kokoro model weights missing)")

    preview_text = text.strip() or DEFAULT_PREVIEW_TEXT
    cache_key = hashlib.sha256(f"{voice_id}|{preview_text}|{speed}".encode()).hexdigest()[:32]
    cache_dir = settings.storage_dir / "cache" / "voice_previews"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.wav"

    if not cache_path.exists():
        try:
            engine.synthesize_chunk_to_file(preview_text, voice_id, speed, cache_path)
        except Exception as e:
            cache_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Could not generate preview for voice '{voice_id}': {e}") from e

    return FileResponse(cache_path, media_type="audio/wav")
