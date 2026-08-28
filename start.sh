#!/usr/bin/env bash
# One-click start: launches the backend + frontend and opens the app in your browser.
# Usage: ./start.sh   (or double-click it in Finder if it's set to open in Terminal)
set -e

cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg not found. Install it first (macOS: brew install ffmpeg)."
  exit 1
fi

if [ ! -f backend/models/kokoro-v1.0.onnx ]; then
  echo "Kokoro model weights not found — downloading (~350MB, one-time)..."
  mkdir -p backend/models
  curl -L -o backend/models/kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
  curl -L -o backend/models/voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
fi

cleanup() {
  echo ""
  echo "Stopping BookSpeech..."
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
  wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null
  exit 0
}
trap cleanup INT TERM

echo "Starting backend..."
(cd backend && uv run uvicorn app.main:app --port 8000) &
BACKEND_PID=$!

echo "Starting frontend..."
(cd frontend && python3 -m http.server 5173) &
FRONTEND_PID=$!

echo "Waiting for backend to come up..."
for i in $(seq 1 30); do
  if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo ""
echo "BookSpeech is running:"
echo "  App:     http://localhost:5173"
echo "  API:     http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop."

if command -v open >/dev/null 2>&1; then
  open "http://localhost:5173"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://localhost:5173"
fi

wait
