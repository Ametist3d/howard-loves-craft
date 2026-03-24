from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class RollResolution(BaseModel):
    investigator: Optional[str] = None
    skill: Optional[str] = None
    target: Optional[int] = None
    roll: Optional[int] = None
    outcome: Optional[str] = None  # critical_success | extreme_success | hard_success | regular_success | failure | fumble | raw_roll
    raw_verdict: Optional[str] = None
    roll_type: str = "skill_check"
    used_luck: bool = False
    luck_spent: int = 0


class CharGenRequest(BaseModel):
    prompt: str
    language: str
    era_context: Optional[str] = None

class ChatResponse(BaseModel):
    narrative: str
    suggested_actions: List[str] = []
    state_updates: Optional[Dict[str, Any]] = None
    roll_resolution: Optional[RollResolution] = None
    image_url: Optional[str] = None
    generation_id: Optional[str] = None 

class StartSessionRequest(BaseModel):
    investigators: List[Dict[str, Any]]
    scenarioType: str
    language: str
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
    physical_description: str
    name: str
    occupation: str
    era_context: Optional[str] = None