import datetime
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# pylint: disable=import-error
from utils.db_session import SessionDB
from utils.combat import is_combat_trigger, get_combat_state

logger = logging.getLogger("keeper_ai.engine")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(_BACKEND_DIR, "data")
SESSIONS_DIR = os.path.join(DATA_DIR, "sessions")
_DEBUG_LOG = os.path.join(DATA_DIR, "prebuilt_debug.log")

os.makedirs(SESSIONS_DIR, exist_ok=True)


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        logger.warning("Invalid integer env %s=%r; using %s", name, raw, default)
        return default


# -----------------------------------------------------------------------------
# Session trace helpers
# -----------------------------------------------------------------------------

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


def _dbg(tag: str, content: str) -> None:
    try:
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 60}\n[{ts}] {tag}\n{'=' * 60}\n{content}\n")
    except Exception as e:
        logger.warning("debug log write failed for tag=%s: %s", tag, e)


# -----------------------------------------------------------------------------
# Shared gameplay helpers
# -----------------------------------------------------------------------------

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
    return (
        first_line.rstrip(".")
        or era_context.rstrip(".")
        or "Investigate the immediate anomaly and identify the first actionable lead"
    )


def _should_refresh_digest(turn_count: int) -> bool:
    """
    Story digest is expensive because it invokes an LLM.

    Defaults are intentionally conservative:
    - do not compress the first few turns
    - then compress every N turns

    Override with:
      KEEPER_DIGEST_START_TURN=8
      KEEPER_DIGEST_EVERY=6
      KEEPER_ENABLE_DIGEST=0
    """
    if not _env_bool("KEEPER_ENABLE_DIGEST", True):
        return False

    start_turn = max(1, _env_int("KEEPER_DIGEST_START_TURN", 8))
    every = max(1, _env_int("KEEPER_DIGEST_EVERY", 6))

    turn = int(turn_count or 0)
    return turn >= start_turn and turn % every == 0


def _is_combat_turn(db: SessionDB, message: str) -> bool:
    if is_combat_trigger(message):
        return True
    combat_state = get_combat_state(db)
    return bool(combat_state.get("active"))


# -----------------------------------------------------------------------------
# Shared runtime state
# -----------------------------------------------------------------------------

active_dbs: Dict[str, SessionDB] = {}
_image_results: Dict[str, Optional[str]] = {}


# -----------------------------------------------------------------------------
# Chroma / embeddings
# -----------------------------------------------------------------------------

emb = None
rules_db = None
scen_db = None


def _load_chroma_db(name: str, persist_subdir: str, embedding_function):
    path = os.path.join(DATA_DIR, persist_subdir)
    if not os.path.isdir(path):
        logger.warning("%s Chroma directory not found: %s", name, path)
        return None

    try:
        return Chroma(
            persist_directory=path,
            embedding_function=embedding_function,
        )
    except Exception as e:
        logger.error("Failed to load %s ChromaDB from %s: %s", name, path, e)
        return None


def _init_chroma() -> None:
    global emb, rules_db, scen_db

    if not _env_bool("KEEPER_LOAD_CHROMA", True):
        logger.info("Chroma loading disabled by KEEPER_LOAD_CHROMA=0")
        return

    model_name = os.getenv("KEEPER_EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    logger.info("Initializing embeddings (%s)...", model_name)

    try:
        emb = HuggingFaceEmbeddings(model_name=model_name)
    except Exception as e:
        logger.error("Failed to initialize embeddings %s: %s", model_name, e)
        emb = None
        return

    rules_db = _load_chroma_db("rules_db", "coc_rules_db", emb)

    # Scenario DB is needed for scenario selection/synthesis, not live chat.
    # Keep it available for engine_session, but allow disabling during debugging.
    if _env_bool("KEEPER_LOAD_SCENARIO_DB", True):
        scen_db = _load_chroma_db("scen_db", "coc_scenario_db", emb)
    else:
        logger.info("Scenario DB loading disabled by KEEPER_LOAD_SCENARIO_DB=0")
        scen_db = None

    if rules_db or scen_db:
        logger.info(
            "ChromaDBs loaded | rules_db=%s | scen_db=%s",
            bool(rules_db),
            bool(scen_db),
        )
    else:
        logger.warning("No ChromaDBs loaded")


_init_chroma()


__all__ = [
    "logger",
    "BASE_DIR",
    "DATA_DIR",
    "SESSIONS_DIR",
    "active_dbs",
    "_image_results",
    "emb",
    "rules_db",
    "scen_db",
    "_trace_session",
    "_trace_session_json",
    "_dbg",
    "_derive_initial_objective",
    "_should_refresh_digest",
    "_is_combat_turn",
]
