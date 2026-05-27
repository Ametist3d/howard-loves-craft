import json
import logging
import os
import secrets
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load .env from project root (two levels up from backend/ when file lives in backend/)
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# pylint: disable=import-error
from utils.schemas import (
    AvatarRequest,
    CharGenRequest,
    ChatRequest,
    StartSessionRequest,
    make_chat_response,
)
from utils.engine import DATA_DIR, _image_results, active_dbs
from utils.engine_chat import handle_chat_logic, stream_chat_logic
from utils.engine_session import generate_character_logic, start_session_logic
from utils.helper_story import generate_avatar_logic

# ─── Config from .env ────────────────────────────────────────────────────────
BE_HOST = os.getenv("BE_HOST", "0.0.0.0")
BE_PORT = int(os.getenv("BE_PORT", "8000"))
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "keeper")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "change_me_please")

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("keeper_ai.api")

# ─── In-memory token store ───────────────────────────────────────────────────
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

def _verify_token(request: Request) -> None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = auth.removeprefix("Bearer ").strip()
    if token not in _active_tokens:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
async def login(req: LoginRequest):
    if req.username != AUTH_USERNAME or req.password != AUTH_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = secrets.token_urlsafe(32)
    _active_tokens[token] = req.username
    logger.info("Auth: login successful for %r", req.username)
    return {"token": token}


@app.post("/api/auth/logout")
async def logout(request: Request):
    auth = request.headers.get("Authorization", "")
    token = auth.removeprefix("Bearer ").strip()
    _active_tokens.pop(token, None)
    return {"ok": True}


# ─── Utility endpoints ───────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "provider": os.getenv("LLM_PROVIDER", "ollama"),
        "images_enabled": os.getenv("KEEPER_ENABLE_SCENE_IMAGES", "1"),
        "digest_enabled": os.getenv("KEEPER_ENABLE_DIGEST", "1"),
    }


@app.get("/api/image-status/{generation_id}")
async def image_status(generation_id: str):
    # This remains public because the frontend polls it using generation_id.
    if generation_id not in _image_results:
        raise HTTPException(status_code=404, detail="Unknown generation_id")

    status = _image_results.get(generation_id)
    if status == "pending":
        return {"ready": False, "image_url": None}

    # One-shot read to prevent unbounded in-memory growth.
    _image_results.pop(generation_id, None)
    return {"ready": True, "image_url": status}


# ─── Generation endpoints ────────────────────────────────────────────────────

@app.post("/api/generate-avatar", dependencies=[Depends(_verify_token)])
async def generate_avatar(req: AvatarRequest):
    try:
        return await generate_avatar_logic(req)
    except Exception as e:
        logger.exception("Avatar Generation Error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/generate-character", dependencies=[Depends(_verify_token)])
async def generate_character(req: CharGenRequest):
    try:
        return await generate_character_logic(req)
    except Exception as e:
        logger.exception("Character Generation Error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/start-session", dependencies=[Depends(_verify_token)])
async def start_session(req: StartSessionRequest):
    try:
        return await start_session_logic(req)
    except Exception as e:
        logger.exception("Session Start Error")
        raise HTTPException(status_code=500, detail=str(e)) from e


# ─── Scenario browser endpoints ──────────────────────────────────────────────

def _scenario_db():
    # Import lazily so disabling KEEPER_LOAD_SCENARIO_DB does not break app startup.
    from utils.engine import scen_db  # pylint: disable=import-error,import-outside-toplevel
    return scen_db


@app.get("/api/scenarios", dependencies=[Depends(_verify_token)])
async def list_scenarios():
    try:
        scen_db = _scenario_db()
        if not scen_db:
            return []

        results = scen_db.get(include=["documents", "metadatas"])
        ids = results.get("ids") or []
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []

        if not ids:
            return []

        grouped: dict[str, dict[str, list]] = defaultdict(lambda: {"chunks": [], "metas": []})

        for doc, meta in zip(documents, metadatas):
            meta = meta or {}
            group_key = str(meta.get("scenario_id") or meta.get("source") or "unknown")
            grouped[group_key]["chunks"].append(doc)
            grouped[group_key]["metas"].append(meta)

        def _pick_title(group_key: str, metas: list[dict]) -> str:
            for key in ("scenario_title", "display_name", "title_en", "title_original", "Header_2"):
                for meta in metas:
                    value = str((meta or {}).get(key, "") or "").strip()
                    if value:
                        return value
            return os.path.basename(group_key).replace(".md", "").replace("_", " ")

        def _pick_theme(metas: list[dict]) -> str:
            values = []
            for key in ("type", "archetype", "role"):
                for meta in metas:
                    value = str((meta or {}).get(key, "") or "").strip()
                    if value:
                        values.append(value)
            unique = []
            seen = set()
            for value in values:
                low = value.lower()
                if low in seen:
                    continue
                seen.add(low)
                unique.append(value)
            return ", ".join(unique[:3])

        def _pick_lang(metas: list[dict]) -> str:
            for meta in metas:
                value = str((meta or {}).get("lang", "") or "").strip()
                if value:
                    return value
            return ""

        def _pick_source_name(group_key: str, metas: list[dict]) -> str:
            for meta in metas:
                value = str((meta or {}).get("scenario_source_name", "") or "").strip()
                if value:
                    return value
            return group_key

        def _load_scenario_content(metas: list[dict], fallback_chunks: list[str]) -> str:
            for meta in metas:
                md_path = str((meta or {}).get("scenario_markdown_path", "") or "").strip()
                if md_path and os.path.exists(md_path):
                    try:
                        return Path(md_path).read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        logger.warning("Failed to read scenario markdown: %s", md_path)
            return "\n\n".join(str(x) for x in fallback_chunks if str(x).strip())

        scenarios = []
        for group_key, data in grouped.items():
            metas = data["metas"]
            scenarios.append({
                "id": group_key,
                "source": _pick_source_name(group_key, metas),
                "title": _pick_title(group_key, metas),
                "theme": _pick_theme(metas),
                "era": _pick_lang(metas),
                "content": _load_scenario_content(metas, data["chunks"]),
            })

        scenarios.sort(key=lambda s: s["title"].lower())
        return scenarios

    except Exception:
        logger.exception("List Scenarios Error")
        return []


@app.get("/api/scenarios/debug", dependencies=[Depends(_verify_token)])
async def debug_scenarios():
    found = []
    backend_root = Path(__file__).resolve().parent

    for root, dirs, files in os.walk(backend_root):
        dirs[:] = [d for d in dirs if d not in {"node_modules", ".git", "__pycache__", "venv", ".venv"}]
        for filename in files:
            lower = filename.lower()
            if filename.endswith((".txt", ".md", ".json")) and ("scenario" in lower or "cthulhu" in lower):
                found.append(str(Path(root) / filename))

    return {"found_files": found, "data_dir": DATA_DIR}


@app.get("/api/session/{session_id}/blueprint", dependencies=[Depends(_verify_token)])
async def get_session_blueprint(session_id: str):
    if session_id not in active_dbs:
        raise HTTPException(status_code=404, detail="Session not found")

    db = active_dbs[session_id]
    row = db.conn.execute(
        "SELECT value FROM kv_store WHERE key='scenario_blueprint_json'"
    ).fetchone()

    if not row or not row["value"]:
        raise HTTPException(status_code=404, detail="No blueprint stored for this session")

    try:
        return json.loads(row["value"])
    except Exception as e:
        raise HTTPException(status_code=500, detail="Stored blueprint is invalid JSON") from e


# ─── Chat endpoints ──────────────────────────────────────────────────────────

@app.post("/api/chat/stream", dependencies=[Depends(_verify_token)])
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        stream_chat_logic(req),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat", dependencies=[Depends(_verify_token)])
async def chat(req: ChatRequest):
    try:
        return await handle_chat_logic(req)
    except Exception:
        logger.exception("Chat Endpoint Error")
        return make_chat_response(
            narrative="(The Keeper's voice distorts... An error occurred in the engine.)",
            suggested_actions=[],
        )


# ─── Runtime settings endpoints ──────────────────────────────────────────────

class ProviderRequest(BaseModel):
    provider: str  # "ollama" | "openai"


@app.post("/api/set-provider", dependencies=[Depends(_verify_token)])
async def set_provider(req: ProviderRequest):
    provider = (req.provider or "").strip().lower()
    if provider not in {"ollama", "openai"}:
        raise HTTPException(status_code=400, detail="provider must be 'ollama' or 'openai'")

    os.environ["LLM_PROVIDER"] = provider
    logger.info("LLM provider switched to: %s", provider)
    return {"ok": True, "provider": provider}


class RuntimeFlagRequest(BaseModel):
    enabled: bool


@app.post("/api/set-scene-images", dependencies=[Depends(_verify_token)])
async def set_scene_images(req: RuntimeFlagRequest):
    os.environ["KEEPER_ENABLE_SCENE_IMAGES"] = "1" if req.enabled else "0"
    return {"ok": True, "enabled": req.enabled}


@app.post("/api/set-story-digest", dependencies=[Depends(_verify_token)])
async def set_story_digest(req: RuntimeFlagRequest):
    os.environ["KEEPER_ENABLE_DIGEST"] = "1" if req.enabled else "0"
    return {"ok": True, "enabled": req.enabled}


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Keeper AI on %s:%s", BE_HOST, BE_PORT)
    uvicorn.run("main:app", host=BE_HOST, port=BE_PORT, reload=True)
