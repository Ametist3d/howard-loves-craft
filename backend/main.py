import os
from collections import defaultdict
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import our custom modules
from utils.schemas import (
    CharGenRequest,
    StartSessionRequest,
    ChatRequest,
    AvatarRequest,
)
from utils.engine import (
    generate_character_logic,
    start_session_logic,
    handle_chat_logic,
    _image_results,
    generate_avatar_logic,
)

# Setup root logger
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("keeper_ai.api")

app = FastAPI(title="Keeper AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/generate-avatar")
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
    # Done — clean up
    del _image_results[generation_id]
    return {"ready": True, "image_url": status}


@app.get("/api/scenarios")
async def list_scenarios():
    try:
        from utils.engine import scen_db

        if not scen_db:
            return []

        # get() needs explicit includes, or use a broad similarity fetch
        results = scen_db.get(include=["documents", "metadatas"])
        if not results["ids"]:
            return []

        # Group chunks by source filename — each .md = one scenario
        from collections import defaultdict

        grouped: dict = defaultdict(lambda: {"chunks": [], "meta": {}})

        for doc, meta in zip(results["documents"], results["metadatas"]):
            source = meta.get("source", "unknown")
            grouped[source]["meta"] = meta
            grouped[source]["chunks"].append(doc)

        scenarios = []
        for source, data in grouped.items():
            meta = data["meta"]
            # Use source filename as title, strip path and extension
            raw_title = os.path.basename(source).replace(".md", "").replace("_", " ")
            scenarios.append(
                {
                    "id": source,
                    "title": raw_title,
                    "theme": meta.get("archetype", ""),
                    "era": meta.get("lang", ""),
                    "content": "\n\n".join(
                        data["chunks"]#[:3]
                    ),  
                }
            )

        scenarios.sort(key=lambda s: s["title"])
        return scenarios

    except Exception as e:
        logger.error(f"List Scenarios Error: {e}")
        return []


@app.post("/api/generate-character")
async def generate_character(req: CharGenRequest):
    try:
        return await generate_character_logic(req)
    except Exception as e:
        logger.error(f"Character Generation Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/start-session")
async def start_session(req: StartSessionRequest):
    try:
        return await start_session_logic(req)
    except Exception as e:
        logger.error(f"Session Start Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scenarios/debug")
async def debug_scenarios():
    import os
    from utils.engine import DATA_DIR, BASE_DIR

    found = []
    search_root = os.path.dirname(BASE_DIR)  # one level up from backend
    for root, dirs, files in os.walk(search_root):
        # skip node_modules, .git, __pycache__
        dirs[:] = [
            d
            for d in dirs
            if d not in ("node_modules", ".git", "__pycache__", "venv", ".venv")
        ]
        for f in files:
            if (
                f.endswith((".txt", ".md", ".json"))
                and "scenario" in f.lower()
                or "cthulhu" in f.lower()
            ):
                found.append(os.path.join(root, f))

    return {"found_files": found, "base_dir": BASE_DIR, "data_dir": DATA_DIR}


@app.get("/api/session/{session_id}/blueprint")
async def get_session_blueprint(session_id: str):
    """Debug endpoint — returns the full generated scenario blueprint."""
    import json
    from utils.engine import active_dbs

    if session_id not in active_dbs:
        raise HTTPException(status_code=404, detail="Session not found")
    db = active_dbs[session_id]
    cur = db.conn.cursor()
    cur.execute("SELECT value FROM kv_store WHERE key='scenario_blueprint_json'")
    row = cur.fetchone()
    if not row:
        raise HTTPException(
            status_code=404, detail="No blueprint stored for this session"
        )
    return json.loads(row["value"])


@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        return await handle_chat_logic(req)
    except Exception as e:
        logger.error(f"Chat Endpoint Error: {e}")
        # Safe fallback so UI doesn't crash
        return {
            "narrative": "(The Keeper's voice distorts... An error occurred in the engine.)",
            "suggested_actions": [],
            "state_updates": None,
        }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting local Keeper AI server on port 8000...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
