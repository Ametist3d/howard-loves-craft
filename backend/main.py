import os
import secrets
import logging
import httpx
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load .env from project root (two levels up from backend/)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from utils.schemas import CharGenRequest, StartSessionRequest, ChatRequest, AvatarRequest
from utils.engine import (
    generate_character_logic,
    start_session_logic,
    handle_chat_logic,
    _image_results,
    stream_chat_logic,
)
from utils.helpers import generate_avatar_logic

# ─── Config from .env ────────────────────────────────────────────────────────
BE_HOST = os.getenv("BE_HOST", "0.0.0.0")
BE_PORT = int(os.getenv("BE_PORT", "8000"))
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "keeper")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "change_me_please")

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("keeper_ai.api")

# ─── In-memory token store ───────────────────────────────────────────────────
# Maps token → username. Single-user app so one active token is fine.
_active_tokens: dict[str, str] = {}

app = FastAPI(title="Keeper AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_static_dir = Path(__file__).resolve().parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ─── Auth helpers ─────────────────────────────────────────────────────────────

def _verify_token(request: Request):
    """Dependency: extracts and validates Bearer token from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth.removeprefix("Bearer ").strip()
    if token not in _active_tokens:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ─── Auth endpoint (no token required) ───────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    if req.username != AUTH_USERNAME or req.password != AUTH_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = secrets.token_urlsafe(32)
    _active_tokens[token] = req.username
    logger.info(f"Auth: login successful for '{req.username}'")
    return {"token": token}

@app.post("/api/auth/logout")
async def logout(request: Request):
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    _active_tokens.pop(token, None)
    return {"ok": True}


# ─── Protected routes ─────────────────────────────────────────────────────────
    
@app.post("/api/generate-avatar", dependencies=[Depends(_verify_token)])
async def generate_avatar(req: AvatarRequest):
    try:
        return await generate_avatar_logic(req)
    except Exception as e:
        logger.error(f"Avatar Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/image-status/{generation_id}")
async def image_status(generation_id: str):
    status = _image_results.get(generation_id)
    if status is None and generation_id not in _image_results:
        raise HTTPException(status_code=404, detail="Unknown generation_id")
    if status == "pending":
        return {"ready": False, "image_url": None}
    del _image_results[generation_id]
    return {"ready": True, "image_url": status}

@app.get("/api/scenarios", dependencies=[Depends(_verify_token)])
async def list_scenarios():
    try:
        from utils.engine import scen_db
        if not scen_db:
            return []

        results = scen_db.get(include=["documents", "metadatas"])
        if not results["ids"]:
            return []

        grouped: dict = defaultdict(lambda: {"chunks": [], "meta": {}})
        for doc, meta in zip(results["documents"], results["metadatas"]):
            source = meta.get("source", "unknown")
            grouped[source]["meta"] = meta
            grouped[source]["chunks"].append(doc)

        scenarios = []
        for source, data in grouped.items():
            meta = data["meta"]
            raw_title = os.path.basename(source).replace(".md", "").replace("_", " ")
            scenarios.append({
                "id": source,
                "title": raw_title,
                "theme": meta.get("archetype", ""),
                "era": meta.get("lang", ""),
                "content": "\n\n".join(data["chunks"]),  # full document
            })

        scenarios.sort(key=lambda s: s["title"])
        return scenarios

    except Exception as e:
        logger.error(f"List Scenarios Error: {e}")
        return []

@app.get("/api/scenarios/debug", dependencies=[Depends(_verify_token)])
async def debug_scenarios():
    from utils.engine import DATA_DIR, BASE_DIR as ENGINE_BASE_DIR
    found = []
    search_root = os.path.dirname(ENGINE_BASE_DIR)
    for root, dirs, files in os.walk(search_root):
        dirs[:] = [d for d in dirs if d not in ("node_modules", ".git", "__pycache__", "venv", ".venv")]
        for f in files:
            if (f.endswith((".txt", ".md", ".json")) and "scenario" in f.lower() or "cthulhu" in f.lower()):
                found.append(os.path.join(root, f))
    return {"found_files": found, "data_dir": DATA_DIR}

@app.post("/api/generate-character", dependencies=[Depends(_verify_token)])
async def generate_character(req: CharGenRequest):
    try:
        return await generate_character_logic(req)
    except Exception as e:
        logger.error(f"Character Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/start-session", dependencies=[Depends(_verify_token)])
async def start_session(req: StartSessionRequest):
    try:
        return await start_session_logic(req)
    except Exception as e:
        logger.error(f"Session Start Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session/{session_id}/blueprint", dependencies=[Depends(_verify_token)])
async def get_session_blueprint(session_id: str):
    import json
    from utils.engine import active_dbs
    if session_id not in active_dbs:
        raise HTTPException(status_code=404, detail="Session not found")
    db = active_dbs[session_id]
    cur = db.conn.cursor()
    cur.execute("SELECT value FROM kv_store WHERE key='scenario_blueprint_json'")
    row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No blueprint stored for this session")
    return json.loads(row["value"])


@app.post("/api/chat/stream", dependencies=[Depends(_verify_token)])
async def chat_stream(req: ChatRequest):
    """SSE endpoint — streams tokens as they are generated."""
    return StreamingResponse(
        stream_chat_logic(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx buffering if behind proxy
        },
    )

@app.post("/api/chat", dependencies=[Depends(_verify_token)])
async def chat(req: ChatRequest):
    try:
        return await handle_chat_logic(req)
    except Exception as e:
        logger.error(f"Chat Endpoint Error: {e}")
        return {
            "narrative": "(The Keeper's voice distorts... An error occurred in the engine.)",
            "suggested_actions": [],
            "state_updates": None,
        }

class ProviderRequest(BaseModel):
    provider: str  # "ollama" | "openai"

@app.post("/api/set-provider", dependencies=[Depends(_verify_token)])
async def set_provider(req: ProviderRequest):
    if req.provider not in ("ollama", "openai"):
        raise HTTPException(status_code=400, detail="provider must be 'ollama' or 'openai'")
    os.environ["LLM_PROVIDER"] = req.provider
    logger.info(f"LLM provider switched to: {req.provider}")
    return {"ok": True, "provider": req.provider}

# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Keeper AI on {BE_HOST}:{BE_PORT}")
    uvicorn.run("main:app", host=BE_HOST, port=BE_PORT, reload=True)
