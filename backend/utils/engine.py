import os
import logging
from typing import Dict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
import datetime
import json
#pylint: disable=import-error
from utils.db_session import SessionDB
from utils.combat import is_combat_trigger, get_combat_state

logger = logging.getLogger("keeper_ai.engine")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(_BACKEND_DIR, "data")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")


def _session_trace_path(db_or_path) -> str:
    if hasattr(db_or_path, "db_path"):
        db_path = db_or_path.db_path
    else:
        db_path = str(db_or_path)

    p = Path(db_path)
    if p.suffix == ".sqlite":
        return str(p.with_suffix(".trace.log"))
    return str(p.parent / f"{p.name}.trace.log")


def _trace_session(db_or_path, tag: str, content: str) -> None:
    try:
        trace_path = _session_trace_path(db_or_path)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 100}\n[{ts}] {tag}\n{'=' * 100}\n{content}\n")
    except Exception as e:
        logger.warning("trace write failed for tag=%s: %s", tag, e)


def _trace_session_json(db_or_path, tag: str, payload) -> None:
    try:
        _trace_session(
            db_or_path,
            tag,
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        )
    except Exception as e:
        logger.warning("trace json write failed for tag=%s: %s", tag, e)

def _is_combat_turn(db: SessionDB, message: str) -> bool:
    if is_combat_trigger(message):
        return True
    combat_state = get_combat_state(db)
    return bool(combat_state.get("active"))

import datetime
_DEBUG_LOG = os.path.join(DATA_DIR, "prebuilt_debug.log")


def _dbg(tag: str, content: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*60}\n[{ts}] {tag}\n{'='*60}\n{content}\n")

os.makedirs(SESSIONS_DIR, exist_ok=True)


def _derive_initial_objective(blueprint: dict, scenario_atoms_text: str, era_context: str) -> str:
    if isinstance(blueprint, dict):
        acts = blueprint.get("acts") or []
        if acts:
            first_act = acts[0]
            scenes = first_act.get("scenes") or []
            if scenes:
                scene = scenes[0]
                trigger = str(scene.get("trigger", "") or "").strip().rstrip(".")
                what_happens = str(scene.get("what_happens", "") or "").strip().rstrip(".")
                dramatic_question = str(scene.get("dramatic_question", "") or "").strip().rstrip(" ?.")

                if trigger:
                    return trigger
                if what_happens:
                    return what_happens
                if dramatic_question:
                    return dramatic_question

        for key in ("inciting_hook", "core_mystery"):
            value = str(blueprint.get(key, "") or "").strip()
            if value:
                return value.rstrip(".")
            
    first_line = (scenario_atoms_text or "").splitlines()[0].strip() if scenario_atoms_text else ""
    return first_line.rstrip(".") or era_context.rstrip(".") or "Investigate the immediate anomaly and identify the first actionable lead"


def _should_refresh_digest(turn_count: int) -> bool:
    # Refresh aggressively early on, then every 2 turns.
    return turn_count <= 3 or turn_count % 2 == 0


active_dbs: Dict[str, SessionDB] = {}
_image_results: Dict[str, str | None] = {}

logger.info("Initializing Embeddings (intfloat/multilingual-e5-large)...")
emb = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

try:
    rules_db = Chroma(persist_directory=os.path.join(DATA_DIR, "coc_rules_db"), embedding_function=emb)
    scen_db = Chroma(persist_directory=os.path.join(DATA_DIR, "coc_scenario_db"), embedding_function=emb)
    logger.info("ChromaDBs loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load ChromaDBs: {e}")
    rules_db, scen_db = None, None


from utils.engine_session import generate_character_logic, generate_opening_scene_logic, start_session_logic
from utils.engine_chat import handle_chat_logic, stream_chat_logic
