# BookSpeech

EPUB → audiobook (`.m4b`) converter. Local-first TTS via Kokoro-82M, FastAPI backend with a background job queue, and a simple local web UI.

## Quick start

```bash
./start.sh
```

Installs nothing you don't already have — it just checks for `uv`/`ffmpeg`, downloads the Kokoro model weights on first run if missing, starts the backend + frontend, and opens the app in your browser. `Ctrl+C` stops both. (Same as `make start`.)

## Architecture

```
EPUB file
   │
   ▼
[Plan 2] epub_parser.py  ──►  Book{title, author, cover, chapters[]}
   │
   ▼
[Plan 3] text_cleaner.py ──►  normalized, TTS-ready chapter text
   │
   ▼
[Plan 4] chunker.py      ──►  ordered list of TTS-safe text chunks
   │
   ▼
[Plan 5] TTSEngine (KokoroEngine) ──► per-chunk wav files
   │
   ▼
[Plan 6] audio_assembler.py ──► chaptered .m4b (ffmpeg + mutagen)
   │
   ▼
[Plan 7] FastAPI job queue/API  ◄──►  [Plan 8/9] Web UI (upload, voice preview, progress, library, settings)
```

Jobs run as background asyncio tasks (see Plan 7) so multi-hour conversions never block HTTP requests. Job/book metadata persists in SQLite (`storage/bookspeech.db`). Failed jobs are resumable — already-synthesized chapters are reused on retry rather than re-run (see Plan 10).

## Repo layout

```
backend/         FastAPI app, TTS engines, pipeline modules (uv-managed Python project)
  app/
    tts/         TTSEngine interface + KokoroEngine
    routers/     API route modules (books, jobs, voices, library, settings)
  Dockerfile
frontend/        Static web UI
storage/
  uploads/       Uploaded epubs + per-job intermediate chunk audio
  output/        Finished .m4b files
  logs/          Per-job log files (<job_id>.log)
  cache/         Cached voice preview audio
scripts/         Manual QA / benchmark / smoke-test scripts
tests/           Pytest suite (parser, cleaner, chunker, assembly, API, hardening)
docker-compose.yml
TROUBLESHOOTING.md
```

## Setup

### Native

1. Install [uv](https://docs.astral.sh/uv/): `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Install `ffmpeg` on your system PATH:
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - Windows: download from ffmpeg.org and add to PATH
3. `make install` — installs backend Python deps (FastAPI, ebooklib, kokoro-onnx, onnxruntime, etc.) via uv
4. Download the Kokoro-82M model weights into `backend/models/`:
   ```bash
   mkdir -p backend/models
   curl -L -o backend/models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
   curl -L -o backend/models/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
   ```
   (~325MB + 28MB, not checked into git). Manual QA: `cd backend && uv run python ../scripts/tts_smoke_test.py`
5. Copy `backend/.env.example` to `backend/.env` and adjust storage paths / defaults if needed
6. `make run-backend` — starts the API at `http://localhost:8000` (health check: `/health`)
7. `make run-frontend` — serves the static UI at `http://localhost:5173`
8. `make run-tests` — runs the pytest suite

### Docker

After downloading the model weights to `backend/models/` (step 4 above):

```bash
docker compose up --build
```

Backend at `http://localhost:8000`, frontend at `http://localhost:5173`. Storage and models are bind-mounted from the host so audiobooks/logs survive a container restart.

## Settings

Default voice and max concurrent jobs are editable at runtime via the UI's Settings tab (or `GET`/`PUT /settings`), persisted to `storage/settings.json` and re-applied on the next startup. Storage/output paths are fixed at process start (env vars only) since changing them live is unsafe mid-job.

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common ffmpeg errors, ONNX Runtime/GPU setup issues, and epub parsing edge cases (including DRM detection).

## Build order

10 plans × 10 sub-plans = 100 tasks. Recommended order: 1 → 2 → 3 → 4 → 5 → 6 (prove the core pipeline end-to-end as a script) → 7 → 8 → 9 → 10. See `bookspeech.html` for the full task breakdown.
