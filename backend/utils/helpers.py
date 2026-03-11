import os
import re
import json
from utils.db_session import SessionDB

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

def read_prompt(filename: str) -> str:
    with open(os.path.join(PROMPTS_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()

def extract_json(text: str) -> dict:
    text = text.strip()
    
    # Remove markdown codeblocks if present
    text = re.sub(r'^```[a-zA-Z]*\n', '', text)
    text = re.sub(r'\n```$', '', text)
    
    # Try to isolate the JSON block
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        text = text[start:end+1]
        
    # PRE-PROCESSOR: Fix LLM outputting "-1d6" instead of integers in JSON
    # This regex changes {"sanity_change": -1d6} into {"sanity_change": -1}
    text = re.sub(r':\s*([-+]?\d+)[dD]\d+', r': \1', text)

    try:
        # strict=False allows some minor control character violations
        return json.loads(text, strict=False)
    except json.JSONDecodeError as e:
        # FALLBACK: If JSON is completely broken (unescaped newlines, missing quotes)
        # We use regex to extract the narrative so the game doesn't crash!
        print(f"JSON Parse Error intercepted: {e}. Attempting Regex Fallback.")
        
        # Look for everything between "narrative": " and the next field or end of string
        narrative_match = re.search(r'"narrative"\s*:\s*"(.*?)"(?:\s*,|\s*\n\s*"suggested_actions")', text, re.DOTALL | re.IGNORECASE)
        
        narrative = narrative_match.group(1) if narrative_match else text
        
        # Clean up the broken string so it renders nicely in React
        narrative = narrative.replace('\n', ' ').replace('\\n', '\n')
        
        return {
            "narrative": narrative + "\n\n*(OOC: The Keeper's connection wavered, some state updates may have been lost.)*",
            "suggested_actions": ["Continue"],
            "state_updates": None
        }

def get_chat_history(db: SessionDB, limit: int = 10) -> str:
    events = db.list_events(limit=limit * 2)
    pairs = []
    for e in reversed(events):
        if e.get("event_type") == "CHAT":
            payload = e.get("payload", {})
            role = payload.get("role", "Unknown")
            content = payload.get("content", "")
            pairs.append((role, content))

    chat_lines = []
    for i, (role, content) in enumerate(pairs):
        if role.lower() == "keeper":
            # Keeper messages are ALWAYS summarised — never full.
            # Full text is a copy-template that Gemma will parrot.
            if len(content) > 150:
                # Keep only the LAST sentence as a context anchor
                last_sentence = content.rstrip().rsplit(".", 1)[-1].strip()
                if not last_sentence or len(last_sentence) < 10:
                    last_sentence = content[-120:].strip()
                content = f"[...сцена описана ранее...] {last_sentence}"
            chat_lines.append(f"KEEPER: {content}")
        else:
            # Player messages stay full — they're short actions
            chat_lines.append(f"USER: {content}")

    return "\n\n".join(chat_lines)

# def get_chat_history(db: SessionDB, limit: int = 10) -> str:
#     events = db.list_events(limit=limit * 2)
#     chat_lines = []
#     for e in reversed(events):
#         if e.get("event_type") == "CHAT":
#             payload = e.get("payload", {})
#             role = payload.get("role", "Unknown")
#             content = payload.get("content", "")
            
#             # Truncate Keeper messages — model must not use them as a copy template.
#             # Show only the last 80 chars as a context anchor, not the full narrative.
#             if role.lower() == "keeper" and len(content) > 300:
#                 content = "...[предыдущая сцена]... " + content[-100:].strip()
            
#             chat_lines.append(f"{role.upper()}: {content}")
#     return "\n\n".join(chat_lines)