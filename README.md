# DocQ&A Lite

Upload a PDF or text file, ask questions, and get the most relevant excerpts using sentence embeddings + FAISS.

## What this project is about
- **Frontend:** Next.js (App Router + Tailwind)
- **Backend:** FastAPI (Python 3.11)
- **AI:** sentence-transformers/all-MiniLM-L6-v2 + FAISS
- **Infra:** Docker (compose), optional Kubernetes manifests

## My individual contribution
Designed and built the entire stack: API (upload, ask, reset), chunking & embeddings, FAISS index, Next.js UI, Dockerfiles, docker-compose, K8s manifests, docs.

## Local dev
```bash
# API
cd backend && uvicorn app:app --reload --port 8000
# Web
cd frontend && npm run dev
# http://localhost:3000