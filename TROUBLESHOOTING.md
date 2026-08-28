# Troubleshooting

## ffmpeg errors

**`ffmpeg failed: ... Option map_metadata (...) cannot be applied to input url`**
An ffmpeg command has an input-only option placed after an output file, or
after a later `-i`. All `-i` flags must come first, then output/global
options. If you're editing `app/audio_assembler.py`, keep `-map_metadata`
and friends after every `-i`.

**`Could not find tag for codec h264 in stream #1, codec not currently supported in container`**
This happens if a cover image gets treated as a video stream to re-encode
instead of copied as-is. Fix: pass `-c:v copy` for the cover art stream (see
`encode_m4b` in `app/audio_assembler.py`) — never let ffmpeg re-encode a
JPEG/PNG cover as h264.

**`ffmpeg: command not found`**
ffmpeg isn't on PATH. Install it:
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`
- Windows: download a build from ffmpeg.org and add its `bin/` to PATH

Verify with `ffmpeg -version`.

**Chapters missing from the output `.m4b`**
Check with `ffprobe -v error -show_chapters out.m4b`. If ffprobe shows
chapters but `mutagen.mp4.MP4(path).chapters` is empty, you're reading them
wrong — use `MP4(path).chapters` directly (a loaded property), not
`MP4Chapters(audio)` (that class needs `.load()` with raw atoms, not an
already-open `MP4` object).

## ONNX Runtime / GPU setup issues

**Synthesis is slow / clearly running on CPU when you expected GPU**
`kokoro-onnx` auto-selects providers via `kokoro_onnx.session.resolve_providers()`:
it only picks up GPU providers if an accelerated onnxruntime distribution
(`onnxruntime-gpu`, `onnxruntime-directml`, etc.) is installed *instead of*
plain `onnxruntime`. Installing both leaves plain CPU in effect. To force a
provider explicitly, set the `ONNX_PROVIDER` env var (e.g.
`ONNX_PROVIDER=CUDAExecutionProvider`) before starting the backend.

**`KokoroEngine` raises `FileNotFoundError` on startup**
The model weights aren't downloaded. See the Setup section in README.md —
you need both `backend/models/kokoro-v1.0.onnx` and
`backend/models/voices-v1.0.bin`.

**Voice preview / job creation returns 503 "TTS engine not loaded"**
Same root cause as above — the backend logs a warning at startup and leaves
`app.state.engine = None` rather than crashing, so the rest of the API stays
usable (uploads still work) while synthesis-dependent endpoints 503.

## EPUB parsing edge cases

**A chapter's title looks wrong or chapters got merged**
The parser detects chapter boundaries by `<h1>/<h2>/<h3>` headings per
spine item; spine items without their own heading are merged into the
previous chapter (many epubs split one logical chapter across multiple
XHTML files). If an epub's headings are inconsistent, you'll see either
over-merging or one giant "chapter" — check `app/epub_parser.py`'s
`_SKIP_ID_PATTERNS` / `_MIN_CHAPTER_CHARS` heuristics if a specific book
misbehaves.

**Upload rejected with "This epub is DRM-protected"**
Real DRM (Adobe ADEPT, etc.) registers `META-INF/encryption.xml` with a
non-font-obfuscation algorithm. Remove DRM first — e.g. with
[Calibre](https://calibre-ebook.com/) plus a DeDRM plugin — then re-upload.
If you get this on a book you're sure isn't DRM-protected, check
`is_drm_protected()` in `app/epub_parser.py`; IDPF font obfuscation
(`http://www.idpf.org/2008/embedding`) is intentionally excluded.

**Upload rejected with "Could not parse epub"**
Usually a genuinely corrupt/truncated zip, or an epub2/epub3 quirk
`ebooklib` doesn't handle. The exact `ebooklib`/zipfile exception is
included in the error detail — check that first.

**"No chapters could be extracted from this epub"**
Every spine item was filtered out as boilerplate (`_MIN_CHAPTER_CHARS`) or
matched a skip pattern (TOC/cover/titlepage). Rare, but can happen on very
short epubs (novellas split into many tiny files) — lower
`_MIN_CHAPTER_CHARS` in `app/epub_parser.py` if you hit this legitimately.

## Job / synthesis failures

**A job failed partway through a long book**
Check its per-job log at `storage/logs/<job_id>.log` for the actual
exception. Retry via `POST /jobs/{id}/retry` (or the UI's Retry button) —
already-synthesized chapters are reused rather than re-run, so you don't
lose the completed work.

**Job fails immediately with a disk space error (HTTP 507)**
The pre-flight check in `app/disk_space.py` estimates ~15 bytes of
intermediate storage per source character (uncompressed wav dwarfs the
final AAC). Free up space or point `STORAGE_DIR` at a larger volume.

**One chunk sounds garbled or silent in the middle of an otherwise fine chapter**
`KokoroEngine.synthesize()` already retries near-silent output up to twice
before giving up and returning it anyway (see `app/tts/kokoro_engine.py`).
Very rare residual garbling is usually a single problematic chunk of text —
check the job log for `Near-silent synthesis output` warnings and consider
adjusting `app/text_cleaner.py` rules if it's a recurring phrase pattern.
