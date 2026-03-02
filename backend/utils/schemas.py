from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class CharGenRequest(BaseModel):
    prompt: str
    language: str
    era_context: Optional[str] = None

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