from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Shared constants
# -----------------------------------------------------------------------------

CHARACTERISTIC_NAMES = ("STR", "CON", "SIZ", "DEX", "APP", "INT", "POW", "EDU")


# -----------------------------------------------------------------------------
# Character generation
# -----------------------------------------------------------------------------

class CharacterSkill(BaseModel):
    name: str = ""
    value: int = 0


class CharacterCharacteristics(BaseModel):
    STR: int = 50
    CON: int = 50
    SIZ: int = 50
    DEX: int = 50
    APP: int = 50
    INT: int = 50
    POW: int = 50
    EDU: int = 50


class CharacterResponse(BaseModel):
    name: str = "Unknown Investigator"
    occupation: str = "Investigator"
    age: int = 30
    residence: str = ""
    birthplace: str = ""
    characteristics: CharacterCharacteristics = Field(default_factory=CharacterCharacteristics)
    skills: List[CharacterSkill] = Field(default_factory=list)
    inventory: List[str] = Field(default_factory=list)
    background: str = ""
    physical_description: str = ""
    skill_plan: Dict[str, Any] = Field(default_factory=dict)


class CharGenRequest(BaseModel):
    prompt: str
    language: str = "en"
    era_context: Optional[str] = None


# -----------------------------------------------------------------------------
# Runtime chat response contract
# -----------------------------------------------------------------------------

class SceneEntities(BaseModel):
    present_named_entities: List[str] = Field(default_factory=list)


class RollRequest(BaseModel):
    required: bool = False
    # Kept as skill_name for frontend/backward compatibility.
    # Value may be a CoC skill name OR a characteristic target such as STR/DEX/INT.
    skill_name: str = ""
    action_text: str = ""
    reason: str = ""


class RollResolution(BaseModel):
    investigator: str = ""
    skill: str = ""
    target: Optional[int] = None
    roll: Optional[int] = None
    # critical_success | extreme_success | hard_success | regular_success | failure | fumble
    outcome: str = ""
    raw_verdict: str = ""
    roll_type: str = "skill_check"
    used_luck: bool = False
    luck_spent: int = 0


class StateUpdates(BaseModel):
    character_name: str = ""
    hp_change: int = 0
    sanity_change: int = 0
    mp_change: int = 0
    inventory_add: str = ""
    inventory_remove: str = ""
    location_name: str = ""
    location_description: str = ""
    clue_found: str = ""
    clue_content: str = ""
    thread_progress: str = ""


class CombatAction(BaseModel):
    start_combat: bool = False
    end_combat: bool = False
    actor_name: str = ""
    target_name: str = ""
    # attack_melee | attack_firearm | maneuver | move | flee | draw_weapon |
    # ready_firearm | assist | delay | use_item | cast | dive_for_cover | other
    action_type: str = ""
    skill_name: str = ""
    weapon_name: str = ""
    weapon_damage: str = ""
    defender_option: str = ""
    shots_fired: int = 0
    bonus_dice: int = 0
    penalty_dice: int = 0


class ChatResponse(BaseModel):
    narrative: str = ""
    suggested_actions: List[str] = Field(default_factory=list)
    state_updates: StateUpdates = Field(default_factory=StateUpdates)
    roll_resolution: Optional[RollResolution] = None
    roll_request: RollRequest = Field(default_factory=RollRequest)
    combat_action: CombatAction = Field(default_factory=CombatAction)
    scene_entities: SceneEntities = Field(default_factory=SceneEntities)
    image_url: Optional[str] = None
    generation_id: Optional[str] = None


# -----------------------------------------------------------------------------
# API request models
# -----------------------------------------------------------------------------

class StartSessionRequest(BaseModel):
    investigators: List[Dict[str, Any]] = Field(default_factory=list)
    scenarioType: str = "custom"
    language: str = "en"
    customPrompt: Optional[str] = None
    themes: Optional[List[str]] = None
    era_context: Optional[str] = None
    picked_seed: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: str = "local_session"
    rag_enabled: bool = True
    top_k: int = 2
    temperature: float = 0.7
    num_ctx: int = 16384


class AvatarRequest(BaseModel):
    physical_description: str = ""
    name: str = ""
    occupation: str = ""
    era_context: Optional[str] = None


# -----------------------------------------------------------------------------
# Pydantic v1/v2 compatibility helpers
# -----------------------------------------------------------------------------

def _model_to_dict(model: BaseModel) -> dict:
    dumper = getattr(model, "model_dump", None)
    if callable(dumper):
        return dumper()
    return model.dict()


def validate_character_response_payload(payload: dict | None) -> dict:
    validator = getattr(CharacterResponse, "model_validate", None)
    if callable(validator):
        return _model_to_dict(CharacterResponse.model_validate(payload or {}))
    return CharacterResponse.parse_obj(payload or {}).dict()


def validate_chat_response_payload(payload: dict | None) -> dict:
    """
    Normalize any partial/intercepted/LLM response into the full ChatResponse shape.

    This is intentionally permissive: validation should fill missing defaults instead
    of crashing the game loop when the LLM omits optional structures.
    """
    validator = getattr(ChatResponse, "model_validate", None)
    if callable(validator):
        return _model_to_dict(ChatResponse.model_validate(payload or {}))
    return ChatResponse.parse_obj(payload or {}).dict()


# -----------------------------------------------------------------------------
# Convenience factories for non-LLM deterministic responses
# -----------------------------------------------------------------------------

def blank_roll_request() -> dict:
    return _model_to_dict(RollRequest())


def blank_state_updates() -> dict:
    return _model_to_dict(StateUpdates())


def blank_combat_action() -> dict:
    return _model_to_dict(CombatAction())


def blank_scene_entities() -> dict:
    return _model_to_dict(SceneEntities())


def make_chat_response(
    *,
    narrative: str = "",
    suggested_actions: list[str] | None = None,
    roll_request: dict | None = None,
    state_updates: dict | None = None,
    combat_action: dict | None = None,
    scene_entities: dict | None = None,
    image_url: str | None = None,
    generation_id: str | None = None,
) -> dict:
    """Build a schema-safe ChatResponse dict without invoking Pydantic at call sites."""
    return validate_chat_response_payload({
        "narrative": narrative,
        "suggested_actions": suggested_actions or [],
        "roll_request": roll_request or blank_roll_request(),
        "state_updates": state_updates or blank_state_updates(),
        "combat_action": combat_action or blank_combat_action(),
        "scene_entities": scene_entities or blank_scene_entities(),
        "image_url": image_url,
        "generation_id": generation_id,
    })
