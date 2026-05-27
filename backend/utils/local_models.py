from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import threading
from pathlib import Path
from typing import Dict, List

import torch
from sentence_transformers import SentenceTransformer, util
from utils.coc_skills import (
    SKILL_GLOSS,
)

logger = logging.getLogger("keeper_ai.local_models")

MODEL_ROOT = Path(__file__).resolve().parent.parent / "data" / "models"
MINILM_MODEL_DIR = MODEL_ROOT / "all-MiniLM-L6-v2"

_FALSE_ENV_VALUES = {"0", "false", "no", "off", "disabled"}

_STOPWORDS = {
    "the", "and", "for", "with", "into", "from", "that", "this", "more", "very",
    "closely", "carefully", "try", "attempt", "about", "immediately", "now", "then",
    "there", "their", "they", "them", "what", "when", "where", "which", "while",
    "before", "after", "through", "because", "around", "inside", "outside",
}

_minilm_lock = threading.Lock()
_minilm_model = None


def _env_enabled(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in _FALSE_ENV_VALUES


def _ensure_minilm():
    global _minilm_model
    if _minilm_model is not None:
        return _minilm_model

    with _minilm_lock:
        if _minilm_model is not None:
            return _minilm_model

        MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        if not MINILM_MODEL_DIR.exists():
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            model.save(str(MINILM_MODEL_DIR))

        _minilm_model = SentenceTransformer(str(MINILM_MODEL_DIR))
        return _minilm_model


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    text = (text or "").lower()
    return any(p in text for p in patterns)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    text = (text or "").lower()
    return any(re.search(p, text) for p in patterns)


def _heuristic_effort(player_text: str, last_keeper: str) -> dict:
    """
    Cheap deterministic effort classifier.

    The old implementation used local DistilGPT as a pseudo-classifier. It was slow,
    unstable, and produced `margin=nan`. Roll/no-roll policy should be deterministic;
    embeddings may help select a target, not decide whether the action deserves a roll.
    """
    text = (player_text or "").strip().lower()
    keeper = (last_keeper or "").strip().lower()

    if not text:
        return {"effort_level": "routine", "reason": "empty action"}

    explicit_roll = (
        r"\broll\b", r"\bskill check\b", r"\bmake a check\b", r"\bmake a roll\b",
        r"\btest my skill\b",
    )
    if _matches_any(text, explicit_roll):
        return {"effort_level": "expert", "reason": "explicit roll/check request"}

    impossible = (
        "impossible", "cannot", "can't", "sealed permanently", "already destroyed",
    )
    if _contains_any(keeper, impossible) and _contains_any(text, ("open", "enter", "force", "use")):
        return {"effort_level": "impossible", "reason": "established barrier prevents this action"}

    risky = (
        "attack", "shoot", "stab", "fight", "grab", "tackle", "jump across", "leap across",
        "run through", "dive", "crawl under", "poison", "fire", "burning", "trap", "unstable",
        "contaminated", "radiation", "acid", "flood", "collapsing", "chase",
    )
    if _contains_any(text, risky):
        return {"effort_level": "risky", "reason": "danger or immediate physical consequence is involved"}

    physical_force_objects = (
        "gate", "door", "hatch", "trapdoor", "grate", "bars", "barrier",
        "panel", "cover", "lid", "cabinet", "drawer", "crate", "lock",
        "latch", "lever", "wheel", "valve", "stone", "boulder", "rubble",
        "debris", "obstruction", "beam", "chain",
    )

    physical_force_verbs = (
        "force", "force open", "break", "break open", "push", "pull",
        "lift", "pry", "pry open", "kick open", "smash", "heave", "bend",
    )

    if _contains_any(text, physical_force_verbs) and _contains_any(text, physical_force_objects):
        return {
            "effort_level": "risky",
            "reason": "physical force is being applied to a resistant object",
        }

    pressure = (
        "interrogate", "press", "pressure", "demand", "threaten", "confront", "coerce",
        "lean on", "push harder", "insist", "order", "make him tell", "make her tell",
        "catch him lying", "catch her lying", "see if he is lying", "see if she is lying",
    )
    if _contains_any(text, pressure):
        return {"effort_level": "opposed", "reason": "the action pressures or contests another person"}

    resistant_context = (
        "evasive", "withholding", "lying", "avoids eye contact", "nervous", "fear", "defensive",
        "hostile", "refuses", "won't answer", "restricted", "administrator", "operator",
        "rehearsed", "too quickly", "blocks access", "not authorized", "denies access",
    )
    if _contains_any(keeper, resistant_context) and _contains_any(
        text,
        ("ask", "question", "talk", "interview", "demand", "request", "show", "access"),
    ):
        return {"effort_level": "opposed", "reason": "the target or institution is already resisting or withholding"}
    
    generic_examine = (
        "examine the object",
        "inspect the object",
        "look at the object",
        "look closely at the object",
        "examine it closely",
        "inspect it closely",
    )

    hidden_or_expert_context = (
        "hidden", "coded", "symbol", "symbols", "glyph", "inscription",
        "mechanism", "circuit", "technical", "degraded", "pattern",
        "anomaly", "ritual", "trap", "dangerous", "unstable", "powered",
    )

    if _contains_any(text, generic_examine) and not _contains_any(text + " " + keeper, hidden_or_expert_context):
        return {
            "effort_level": "careful",
            "reason": "ordinary close examination of a visible object",
        }
    
    expert = (
        "analyze", "diagnose", "decode", "decipher", "interpret", "translate", "reconstruct",
        "determine what it means", "determine the purpose", "figure out what it does",
        "identify the pattern", "compare the pattern", "compare the dates", "compare the handwriting",
        "trace the source", "look for anomalies", "check for anomalies", "hidden meaning",
        "run a sample", "test the sample", "inspect the mechanism", "examine the circuit",
    )
    if _contains_any(text, expert):
        return {"effort_level": "expert", "reason": "specialist interpretation or hidden pattern analysis is requested"}

    # Basic accessible actions should pass to the Keeper without a roll.
    routine_or_careful = (
        "look", "inspect", "examine", "read", "take", "pick up", "ask", "question", "talk",
        "walk", "go", "enter", "leave", "move", "open", "listen", "wait",
    )
    if _contains_any(text, routine_or_careful):
        return {"effort_level": "careful", "reason": "ordinary accessible interaction"}

    return {"effort_level": "careful", "reason": "local heuristic default"}


async def infer_effort_level_async(
    *,
    player_text: str,
    current_objective: str,
    current_scene: str,
    current_location: str,
    last_keeper: str,
) -> dict:
    # Keep async API for helper_actions compatibility; no model call needed.
    return _heuristic_effort(player_text, last_keeper)


def _token_set(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"[a-zA-Z][a-zA-Z()/-]{2,}", (text or "").lower())
        if t not in _STOPWORDS
    }


def _skill_text(skill_name: str) -> str:
    gloss = SKILL_GLOSS.get(skill_name, "")
    return f"{skill_name}. {gloss}".strip()


def _lexical_score(player_text: str, skill_name: str) -> float:
    action = (player_text or "").lower()
    action_tokens = _token_set(action)
    target_tokens = _token_set(_skill_text(skill_name))
    if not action_tokens or not target_tokens:
        return 0.0

    score = float(len(action_tokens & target_tokens))

    skill_lower = skill_name.lower()
    if re.search(r"(?<![a-z])" + re.escape(skill_lower) + r"(?![a-z])", action):
        score += 4.0

    anchors = {
        "Listen": ("listen", "hear", "overhear", "eavesdrop"),
        "Spot Hidden": ("spot", "notice", "search", "look for", "visual", "scan the room"),
        "Stealth": ("sneak", "hide", "quietly", "unseen", "avoid being seen"),
        "Psychology": ("read motive", "lying", "nervous", "fear", "emotion", "intent"),
        "Intimidate": ("threaten", "intimidate", "coerce", "force", "demand", "order"),
        "Persuade": ("persuade", "convince", "reason with", "negotiate"),
        "Fast Talk": ("bluff", "lie", "trick", "fast talk"),
        "Charm": ("charm", "befriend", "rapport"),
        "Library Use": ("records", "archives", "research", "files", "ledger", "manifest"),
        "Locksmith": ("lock", "locked", "safe", "pick the lock"),
        "Mechanical Repair": ("machine", "mechanism", "gears", "engine", "repair"),
        "Electrical Repair": ("wiring", "circuit", "battery", "radio", "transmitter", "power"),
        "Language (Other)": ("translate", "decipher", "foreign language", "unknown writing", "runes"),
        "Archaeology": ("artifact", "ceramic", "pottery", "ancient", "ruin"),
        "Occult": ("ritual", "occult", "cult", "supernatural", "myth"),
        "Medicine": ("diagnose", "symptom", "disease", "poison", "sedation", "patient"),
        "First Aid": ("bandage", "bleeding", "stabilize", "wound"),
        "Drive Auto": ("drive", "car", "truck", "vehicle", "road"),
        "Pilot": ("pilot", "aircraft", "ship controls", "boat controls"),
        "Dodge": ("dodge", "avoid attack", "evade"),
        "STR": ("force", "lift", "break", "push", "pull", "hold"),
        "CON": ("resist", "endure", "poison", "fatigue", "cold", "heat"),
        "DEX": ("balance", "reflex", "quick", "precision"),
        "INT": ("deduce", "logic", "figure out", "connect facts"),
        "POW": ("willpower", "resist mental", "psychic"),
        "EDU": ("remember", "know", "academic", "education"),
    }
    for phrase in anchors.get(skill_name, ()):  # exact phrase anchors
        if phrase in action:
            score += 5.0

    return score


def _lexical_shortlist(player_text: str, available_skills: List[str], top_k: int) -> List[str]:
    scored = [(skill, _lexical_score(player_text, skill)) for skill in available_skills]
    scored.sort(key=lambda item: item[1], reverse=True)

    strong = [skill for skill, score in scored if score >= 1.0]
    if strong:
        return strong[:top_k]

    # No strong lexical evidence. Return a stable prefix instead of pretending confidence.
    # helper_actions domain pools handle most obvious cases before this function is called.
    return list(available_skills[:top_k])


def _semantic_skill_shortlist_sync(
    *,
    player_text: str,
    current_objective: str,
    current_scene: str,
    current_location: str,
    last_keeper: str,
    available_skills: List[str],
    top_k: int = 6,
) -> List[str]:
    if not available_skills:
        return []

    lexical = _lexical_shortlist(player_text, available_skills, top_k)

    if not _env_enabled("KEEPER_USE_MINILM_SKILL_SHORTLIST", "1"):
        return lexical

    try:
        model = _ensure_minilm()
        query = (
            f"player action: {player_text}\n"
            f"objective: {current_objective}\n"
            f"scene: {current_scene}\n"
            f"location: {current_location}\n"
            f"keeper context: {last_keeper[:800]}"
        )
        skill_texts = [_skill_text(skill) for skill in available_skills]
        embeddings = model.encode(
            [query] + skill_texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        query_emb = embeddings[0]
        skill_embs = embeddings[1:]
        scores = util.cos_sim(query_emb, skill_embs)[0]
        scores = torch.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)

        ranked_idx = torch.argsort(scores, descending=True).tolist()
        semantic: List[str] = []
        for idx in ranked_idx:
            skill = available_skills[idx]
            if skill not in semantic:
                semantic.append(skill)
            if len(semantic) >= top_k:
                break

        # Lexical hits are safer than embedding guesses; keep them first.
        merged: List[str] = []
        for skill in lexical + semantic:
            if skill in available_skills and skill not in merged:
                merged.append(skill)
            if len(merged) >= top_k:
                break
        return merged

    except Exception as exc:
        logger.warning("MiniLM shortlist failed; using lexical shortlist: %s", exc)
        return lexical


async def semantic_skill_shortlist_async(
    *,
    player_text: str,
    current_objective: str,
    current_scene: str,
    current_location: str,
    last_keeper: str,
    available_skills: List[str],
    top_k: int = 6,
) -> List[str]:
    return await asyncio.to_thread(
        _semantic_skill_shortlist_sync,
        player_text=player_text,
        current_objective=current_objective,
        current_scene=current_scene,
        current_location=current_location,
        last_keeper=last_keeper,
        available_skills=available_skills,
        top_k=top_k,
    )
