.PHONY: start run-backend run-frontend run-tests install

start:
	./start.sh

install:
	cd backend && uv sync

run-backend:
	cd backend && uv run uvicorn app.main:app --reload --port 8000

run-frontend:
	cd frontend && python3 -m http.server 5173

run-tests:
	cd backend && uv run pytest ../tests -v
