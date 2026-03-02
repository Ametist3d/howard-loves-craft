# 🐙 Call of Cthulhu — AI Keeper

A digital tabletop RPG application powered by a local LLM that acts as an automated Game Master (Keeper) for the Call of Cthulhu 7th Edition ruleset. Generate scenarios, create investigators, roll dice, and experience Lovecraftian horror narratives — all driven by AI running entirely on your machine.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Gemma 3:27b via Ollama (local) |
| Backend | Python · FastAPI · LangChain |
| Vector DB | ChromaDB (RAG for rules & scenarios) |
| Session DB | SQLite (per-session game state) |
| Embeddings | `intfloat/multilingual-e5-large` (HuggingFace) |
| Frontend | React · TypeScript · Vite |

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- ~20 GB free disk space (for Gemma 3:27b model weights)
- 16+ GB RAM recommended (32 GB for comfortable inference)

---

## 1. Ollama Setup

### Install Ollama

**Linux / macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** Download the installer from [ollama.com](https://ollama.com/download)

### Pull the Model

```bash
ollama pull gemma3:27b
```

> ⏳ This downloads ~17 GB. Grab a coffee.

### Verify It Works

```bash
ollama run gemma3:27b "Say something eldritch."
```

### Start Ollama Server

Ollama runs as a background service automatically after install. If you need to start it manually:

```bash
ollama serve
```

The API will be available at `http://localhost:11434`. The backend expects it there — no config needed.

---

## 2. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Ingest Rulebook & Scenarios into ChromaDB

```bash
# Ingest Call of Cthulhu rules
python data/ingest_rules.py

# Ingest scenario atoms
python data/ingest_scenarios.py
```

> This populates `backend/data/coc_rules_db/` and `backend/data/coc_scenario_db/`.

### Start the Backend

```bash
python main.py
```

Backend runs at `http://localhost:8000`.

---

## 3. Frontend Setup

```bash
cd frontend

npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

---

## 4. Project Structure

```
CALL-OF-C/
├── backend/
│   ├── data/
│   │   ├── coc_rules_db/        # ChromaDB — rulebook vectors
│   │   ├── coc_scenario_db/     # ChromaDB — scenario atoms
│   │   └── sessions/            # SQLite session files (gitignored)
│   ├── db/
│   │   └── session_manager.py
│   ├── prompts/
│   │   ├── character_gen.txt    # Character generation prompt
│   │   ├── keeper_chat.txt      # Main Keeper narration prompt
│   │   └── scenario_gen.txt     # Scenario synthesis prompt
│   ├── utils/
│   │   ├── db_session.py        # SQLite session DB logic
│   │   ├── engine.py            # LLM chains, RAG, game logic
│   │   ├── helpers.py
│   │   └── schemas.py           # Pydantic request/response models
│   └── main.py                  # FastAPI entry point
├── frontend/
│   ├── components/
│   │   ├── CharacterSheet.tsx
│   │   ├── ChatInterface.tsx
│   │   ├── DiceRoller.tsx
│   │   ├── SetupScreen.tsx
│   │   └── SettingsModal.tsx
│   ├── services/
│   │   └── apiService.ts
│   ├── App.tsx
│   └── index.tsx
└── requirements.txt
```

---

## 5. API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/scenarios` | List available prebuilt scenarios |
| `POST` | `/api/generate-character` | Generate an investigator |
| `POST` | `/api/start-session` | Initialize a new game session |
| `POST` | `/api/chat` | Send player action, receive Keeper response |
| `GET` | `/api/scenarios/debug` | Debug scenario file discovery |

---

## 6. Configuration

The backend reads from `backend/.env` (create it if it doesn't exist):

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:27b
```

Default values are hardcoded in `engine.py` so this file is optional for local dev.

---

## 7. Gameplay Flow

1. **Setup** — Choose or generate investigators (character creation via AI)
2. **Scenario** — Pick a prebuilt scenario, generate a random one, or write a custom prompt
3. **Session** — The AI Keeper narrates the story, reacts to your actions, and manages game state
4. **Skill Checks** — The DiceRoller component handles rolls; results are sent back to the Keeper for resolution
5. **State Tracking** — HP, Sanity, MP, clues, locations, and NPCs are persisted per session in SQLite

---

## Troubleshooting

**Ollama not responding:**
```bash
# Check if the service is running
curl http://localhost:11434/api/tags

# Restart
ollama serve
```

**ChromaDB empty / no scenarios loading:**
```bash
# Re-run ingestion scripts from the backend/ directory
python data/ingest_scenarios.py
```

**Slow inference:**
Gemma 3:27b is large. If responses are too slow, you can switch to a smaller model by changing `model="gemma3:27b"` to `model="gemma3:12b"` in `backend/utils/engine.py` (three occurrences) and pulling it first:
```bash
ollama pull gemma3:12b
```

**HuggingFace embeddings downloading on first run:**
The `intfloat/multilingual-e5-large` model (~560 MB) is downloaded automatically on first backend start. This is expected.

---

## Language Support

The Keeper narrates in the language configured per session. Russian (`ru`) is the default, with Lovecraftian/Anglo-Saxon naming conventions preserved for proper nouns, locations, and artifacts.

---

## License

For personal and educational use. Call of Cthulhu is a registered trademark of Chaosium Inc.
