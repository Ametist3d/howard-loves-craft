import json
import logging
import os
import random
import re

from langchain_core.prompts import PromptTemplate

from utils.db_session import SessionDB, create_session_db_file
from utils.schemas import CharGenRequest, StartSessionRequest
from utils.helpers import (
    extract_blueprint_json,
    extract_json,
    get_language_name,
    get_llm,
    localize_opening_result_fields,
    normalize_language_code,
    read_prompt,
)
from utils.helper_actions import clear_pending_roll, save_pending_roll
from utils.helper_state import (
    apply_state_updates,
    assemble_keeper_prompt,
    build_authoritative_context,
    build_opening_fallback_result,
    looks_like_valid_keeper_response,
    validate_opening_scene_response,
)
from utils.prompt_translate import ensure_translated_prompts
from utils.engine import (
    DATA_DIR,
    SESSIONS_DIR,
    _dbg,
    _derive_initial_objective,
    active_dbs,
    logger,
    scen_db,
)


async def generate_character_logic(req: CharGenRequest) -> dict:
    session_id = "local_session"

    lang = normalize_language_code(req.language)
    prompt_dir = ensure_translated_prompts(session_id, lang)

    llm = get_llm(temperature=0.7)
    chain = PromptTemplate.from_template(
        read_prompt("character_gen.txt", prompt_dir=prompt_dir)
    ) | llm

    response_text = chain.invoke({
        "language": lang,
        "language_name": get_language_name(lang),
        "prompt": req.prompt,
        "era_context": req.era_context or "1920s Lovecraftian Horror — default if no setting selected",
    })

    
    return extract_json(response_text)


async def generate_opening_scene_logic(db: SessionDB, session_id: str = "local_session") -> dict:
    cur = db.conn.cursor()

    cur.execute("SELECT value FROM kv_store WHERE key='scenario_atoms'")
    row_atoms = cur.fetchone()
    campaign_atoms = row_atoms["value"] if row_atoms else ""

    cur.execute("SELECT value FROM kv_store WHERE key='scenario_themes'")
    row_themes = cur.fetchone()
    themes = row_themes["value"] if row_themes else "STANDARD"

    cur.execute("SELECT value FROM kv_store WHERE key='scenario_setting'")
    row_setting = cur.fetchone()
    setting_override = row_setting["value"] if row_setting else "Lovecraftian Horror Lore"

    cur.execute("SELECT value FROM kv_store WHERE key='era_context'")
    row_era = cur.fetchone()
    era_override = row_era["value"] if row_era else "1920s Lovecraftian Horror"

    cur.execute("SELECT value FROM kv_store WHERE key='language'")
    row_lang = cur.fetchone()
    session_language = normalize_language_code(row_lang["value"] if row_lang else "en")

    cur.execute("SELECT value FROM kv_store WHERE key='prompt_dir'")
    row_prompt_dir = cur.fetchone()
    prompt_dir = row_prompt_dir["value"] if row_prompt_dir else None

    context_str, state_str = build_authoritative_context(
        db,
        campaign_atoms=campaign_atoms,
        themes=themes,
    )

    keeper_system_prompt = assemble_keeper_prompt(
        include_roll_resolution=False,
        include_scene_progression=False,
        include_opening_scene=True,
        prompt_dir=prompt_dir,
    )

    llm = get_llm(temperature=0.2)
    chain = PromptTemplate.from_template(keeper_system_prompt) | llm

    opening_action = (
        "Open the scenario with the first playable scene only. "
        "Anchor the investigators in the current scene and current scene location. "
        "Do not skip ahead. "
        "Do not require a roll unless immediate uncertainty is already established. "
        "Provide exactly three actionable suggested actions."
    )

    prompt_vars = {
        "language": session_language,
        "language_name": get_language_name(session_language),
        "campaign_context": (
            context_str
            + "\n\n--- CURRENT GAME STATE ---\n"
            + state_str
        ),
        "era_context": setting_override + " " + era_override,
        "history": "",
        "action": opening_action,
        "last_turn_ban": "",
    }

    response_text = chain.invoke(prompt_vars)

    logger.info(
        "generate_opening_scene_logic(): raw LLM response preview: %r",
        str(response_text)[:4000]
    )

    result = extract_json(response_text)


    state_updates = dict(result.get("state_updates") or {})
    state_updates["thread_progress"] = ""
    result["state_updates"] = state_updates

    # One-shot contract repair only for malformed/non-contract opening replies
    if not looks_like_valid_keeper_response(str(response_text), result):
        logger.warning("Opening scene contract failure; attempting one repair regeneration")

        repair_prompt = (
            "Your previous reply violated the required Keeper output contract.\n"
            "Return ONLY a valid <SYSTEM_RESPONSE_JSON>...</SYSTEM_RESPONSE_JSON> block.\n"
            "Do not ask the user for clarification.\n"
            "Do not explain your answer.\n"
            "Do not output markdown.\n"
            "Preserve the same opening scene, current scene, and current game state.\n"
        )

        repaired_vars = dict(prompt_vars)
        repaired_vars["campaign_context"] = (
            context_str
            + "\n\n--- CURRENT GAME STATE ---\n"
            + state_str
            + "\n\n--- CONTRACT REPAIR NOTICE ---\n"
            + repair_prompt
        )

        repaired_text = chain.invoke(repaired_vars)

        logger.warning("Opening repair raw preview: %r", str(repaired_text)[:2000])

        repaired_result = extract_json(repaired_text)

        state_updates = dict(repaired_result.get("state_updates") or {})
        state_updates["thread_progress"] = ""
        repaired_result["state_updates"] = state_updates

        if looks_like_valid_keeper_response(str(repaired_text), repaired_result):
            result = repaired_result

    # Opening-specific validator, lighter than normal turn validator
    opening_violations = validate_opening_scene_response(db, result)
    if opening_violations:
        logger.warning("Opening scene validation failed: %s", opening_violations)
        db.log_event("OPENING_VALIDATION_FAIL", {
            "violations": opening_violations,
            "narrative": result.get("narrative", "")[:500]
        })

        # Deterministic fallback in English first
        result = build_opening_fallback_result(db)

        # Then translate only user-facing fields to selected session language
        result = localize_opening_result_fields(result, session_language)

    rr = result.get("roll_request") or {}
    if rr.get("required"):
        save_pending_roll(db, rr)
    else:
        clear_pending_roll(db)

    # Opening is system-generated, so do not create a fake User chat message
    db.log_event("SYS_OPENING_SCENE", {"content": "Opening scene generated"})
    db.log_event("CHAT", {"role": "Keeper", "content": result.get("narrative", "")})

    cur.execute("SELECT value FROM kv_store WHERE key='turn_count'")
    row_tc = cur.fetchone()
    turn_count = int(row_tc["value"]) + 1 if row_tc else 1
    cur.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES ('turn_count', ?)",
        (str(turn_count),)
    )
    db.conn.commit()

    apply_state_updates(db, result)
    return result


async def start_session_logic(req: StartSessionRequest) -> dict:
    session_id = "local_session"

    lang = normalize_language_code(req.language)
    prompt_dir = ensure_translated_prompts(session_id, lang)

    # 1. Format the setting string
    themes_str = ', '.join(req.themes).upper() if req.themes else "STANDARD"
    era_context = req.era_context or "Cosmic Horror — derive era and aesthetics from the scenario atoms."

    if req.scenarioType == 'custom' and req.customPrompt:
        setting_desc = f"Custom: {req.customPrompt[:50]}..."
        query_text = req.customPrompt
    elif req.scenarioType == 'prebuilt' and req.picked_seed:
        setting_desc = f"Prebuilt: {req.picked_seed[:50]}..."
        query_text = req.picked_seed
        era_context = req.era_context or era_context
    elif req.picked_seed:
        setting_desc = f"Themes: {themes_str}"
        query_text = req.picked_seed
    else:
        setting_desc = req.scenarioType
        query_text = "Lovecraftian cosmic horror mystery hook and secrets"

    info = create_session_db_file(SESSIONS_DIR, "Call of Cthulhu", setting_desc)
    db = SessionDB(info.db_path)
    active_dbs[session_id] = db
    logger.info(f"Created new unique session DB: {info.db_path}")

    # 2. Insert Investigators
    for inv in req.investigators:
        chars = inv.get("characteristics", {})
        attrs = inv.get("attributes", {})
        aid = db.upsert_actor(
            kind="PC",
            name=inv.get("name", "Unknown"),
            description=inv.get("occupation", ""),
            hp=attrs.get("HP", {}).get("current", 10),
            mp=attrs.get("MagicPoints", {}).get("current", 10),
            san=attrs.get("Sanity", {}).get("current", 50),
            stats={
                "str": chars.get("STR", 50), "con": chars.get("CON", 50), "dex": chars.get("DEX", 50),
                "int": chars.get("INT", 50), "pow": chars.get("POW", 50), "app": chars.get("APP", 50),
                "siz": chars.get("SIZ", 50), "edu": chars.get("EDU", 50)
            },
            notes=inv.get("background", "")
        )
        for skill in inv.get("skills", []):
            db.set_skill(aid, skill["name"], skill["value"])
        db.log_event("SYS_INIT", {"note": f"Character {inv.get('name')} registered."})

    # 3. Build scenario blueprint
    scenario_atoms_text = ""
    blueprint = {}
    lang = normalize_language_code(lang)

    if req.scenarioType == 'prebuilt' and req.picked_seed:
        # ── PREBUILT PATH ──────────────────────────────────────────────────────
        logger.info("Prebuilt scenario selected — extracting blueprint from source content.")
        scenario_source_text = req.picked_seed

        extract_prompt = (
            "You are a Call of Cthulhu scenario analyst. "
            "Below is the full text of a published scenario. "
            "Extract its structure into JSON. Do NOT invent anything — only use what is in the text.\n\n"
            "SCENARIO TEXT:\n"
            f"{scenario_source_text[:6000]}\n\n"
            "Return ONLY a JSON object with these keys:\n"
            "  title (str), era_and_setting (str), inciting_hook (str), core_mystery (str),\n"
            "  key_npc (str), hidden_threat (str), atmosphere_notes (str),\n"
            "  plot_twists (list[str]),\n"
            "  locations (list of {name, description, tags}),\n"
            "  npcs (list of {name, description, role, secret}),\n"
            "  clues (list of {title, content, location}),\n"
            "  plot_threads (list of {name, stakes, steps})\n"
            "JSON:"
        )
        try:
            extract_llm = get_llm(temperature=0.1)
            extract_raw = extract_llm.invoke(extract_prompt)
            _dbg("EXTRACT PROMPT", extract_prompt[:2000])
            _dbg("EXTRACT RAW RESPONSE", str(extract_raw))
            logger.info(f"[PREBUILT] Raw extract response (first 300): {str(extract_raw)[:300]}")
            blueprint = extract_blueprint_json(extract_raw)
            logger.info(f"[PREBUILT] blueprint keys={list(blueprint.keys()) if isinstance(blueprint, dict) else 'NOT A DICT'}")

            scenario_atoms_text = (
                f"SCENARIO TITLE: {blueprint.get('title', 'Unknown')}\n"
                f"SETTING: {blueprint.get('era_and_setting', '')}\n"
                f"HOOK: {blueprint.get('inciting_hook', '')}\n"
                f"KEY NPC: {blueprint.get('key_npc', '')}\n"
                f"ATMOSPHERE: {blueprint.get('atmosphere_notes', '')}"
            )
            logger.info(f"Prebuilt blueprint extracted: {blueprint.get('title', '?')}")

            if lang != "en":
                language_name = get_language_name(lang)
                logger.info(f"[PREBUILT] Translating atoms to {language_name}...")
                translate_prompt = (
                    f"Translate the following text into {language_name}. "
                    "Do NOT translate proper nouns (person names, place names, ship names, artifact names). "
                    "Output ONLY the translated text, nothing else:\n\n"
                    f"{scenario_atoms_text}"
                )
                try:
                    translated_atoms = get_llm(temperature=0.1).invoke(translate_prompt).strip()
                    _dbg("TRANSLATE PROMPT", translate_prompt)
                    _dbg("TRANSLATE RAW RESPONSE", translated_atoms)
                    if translated_atoms:
                        scenario_atoms_text = translated_atoms
                    logger.info(f"[PREBUILT] scenario_atoms_text after translation: {scenario_atoms_text[:200]}")
                except Exception as te:
                    logger.warning(f"Atoms translation failed ({te}), keeping English.")

                for loc in blueprint.get('locations', []):
                    if loc.get('description'):
                        field_prompt = f"Translate to {language_name}, keep proper nouns unchanged, output only the translation:\n{loc['description']}"
                        raw = get_llm(temperature=0.1).invoke(field_prompt).strip()
                        _dbg(f"FIELD location.description [{loc['name']}]", f"IN: {loc['description']}\nOUT: {raw}")
                        loc['description'] = raw or loc['description']

                for npc in blueprint.get('npcs', []):
                    for field in ('description', 'secret'):
                        if npc.get(field):
                            field_prompt = f"Translate to {language_name}, keep proper nouns unchanged, output only the translation:\n{npc[field]}"
                            raw = get_llm(temperature=0.1).invoke(field_prompt).strip()
                            _dbg(f"FIELD npc.{field} [{npc['name']}]", f"IN: {npc[field]}\nOUT: {raw}")
                            npc[field] = raw or npc[field]

                for i, clue in enumerate(blueprint.get('clues', [])):
                    if clue.get('content'):
                        field_prompt = f"Translate to {language_name}, keep proper nouns unchanged, output only the translation:\n{clue['content']}"
                        raw = get_llm(temperature=0.1).invoke(field_prompt).strip()
                        _dbg(f"FIELD clue.content [{clue.get('title', i)}]", f"IN: {clue['content']}\nOUT: {raw}")
                        clue['content'] = raw or clue['content']

                for thread in blueprint.get('plot_threads', []):
                    if thread.get('stakes'):
                        field_prompt = f"Translate to {language_name}, keep proper nouns unchanged, output only the translation:\n{thread['stakes']}"
                        raw = get_llm(temperature=0.1).invoke(field_prompt).strip()
                        _dbg(f"FIELD thread.stakes [{thread['name']}]", f"IN: {thread['stakes']}\nOUT: {raw}")
                        thread['stakes'] = raw or thread['stakes']

        except Exception as e:
            logger.warning(f"Prebuilt extraction failed ({e}), using raw scenario text.")
            blueprint = {}
            scenario_atoms_text = scenario_source_text[:4000]

    elif scen_db:
        # ── RANDOM / CUSTOM PATH ───────────────────────────────────────────────
        logger.info(f"Querying Scenario DB for starting atoms using: '{query_text}'")
        # scen_docs = scen_db.similarity_search(query_text, k=15)

        def _build_scenario_search_queries(seed: str) -> list[str]:
            seed = (seed or "").strip()
            return [
                seed,
                f"{seed}\nFocus on the central uncanny phenomenon, the social context, and the first investigation scene.",
                f"{seed}\nFind scenario fragments with the same type of institutional or social anomaly.",
                f"{seed}\nPrefer clue sources, authority mismatch, procedural wrongness, hidden organizer, and escalating dread.",
            ]


        def _is_useful_atom(doc) -> bool:
            text = (doc.page_content or "").strip()
            meta = getattr(doc, "metadata", {}) or {}
            role = str(meta.get("role", "") or "").lower()
            abstraction = str(meta.get("abstraction", "") or "").lower()
            atom_type = str(meta.get("type", "") or "").lower()
            title_en = str(meta.get("title_en", "") or "").lower()
            display_name = str(meta.get("display_name", "") or "").lower()

            signal_text = " ".join(x for x in [text, abstraction, title_en, display_name] if x).strip()
            if len(signal_text) < 80:
                return False
            if text.lower().startswith("(implied from context"):
                return False
            if atom_type in {"appendix_character", "timeline"} and len(signal_text) < 140:
                return False
            if role.startswith("worldbuilding") or role.startswith("atmosphere"):
                if "anomaly" not in abstraction and "discovery" not in abstraction and "mystery" not in abstraction:
                    return False
            return True


        def _dedupe_docs(docs):
            seen = set()
            out = []
            for doc in docs:
                meta = getattr(doc, "metadata", {}) or {}
                key = (
                    meta.get("source", ""),
                    meta.get("display_name", "") or meta.get("title_en", "") or meta.get("Header_2", ""),
                    meta.get("type", ""),
                    meta.get("role", ""),
                    (doc.page_content or "").strip()[:180],
                )
                if key in seen:
                    continue
                seen.add(key)
                out.append(doc)
            return out


        candidate_docs = []
        for q in _build_scenario_search_queries(query_text):
            candidate_docs.extend(scen_db.similarity_search(q, k=6))

        logger.info("[SCENARIO_DB] Retrieved %d raw candidate docs for query: %r", len(candidate_docs), query_text)
        for i, doc in enumerate(candidate_docs[:12], start=1):
            logger.info(
                "[SCENARIO_DB] candidate_%d meta=%s preview=%r",
                i,
                getattr(doc, "metadata", {}),
                doc.page_content[:220]
            )

        def _seed_terms(seed: str) -> list[str]:
            seed = (seed or "").lower()
            words = re.findall(r"[a-zA-Z0-9']+", seed)
            stop = {
                "the", "a", "an", "and", "or", "of", "to", "for", "with", "in", "on",
                "is", "are", "was", "were", "that", "this", "it", "they", "them",
                "into", "from", "their", "have", "has", "had", "will", "would",
                "local", "ancient", "strange", "mysterious"
            }
            terms = [w for w in words if len(w) >= 4 and w not in stop]
            return terms[:12]


        scen_docs = _dedupe_docs(candidate_docs)
        scen_docs = [d for d in scen_docs if _is_useful_atom(d)]

        seed_terms = _seed_terms(query_text)
        rescored = []

        for doc in scen_docs:
            text = (doc.page_content or "").strip().lower()
            meta = getattr(doc, "metadata", {}) or {}
            abstraction = str(meta.get("abstraction", "") or "").lower()
            role = str(meta.get("role", "") or "").lower()
            header = str(meta.get("Header_2", "") or "").lower()
            atom_type = str(meta.get("type", "") or "").lower()
            archetype = str(meta.get("archetype", "") or "").lower()
            title_en = str(meta.get("title_en", "") or "").lower()
            display_name = str(meta.get("display_name", "") or "").lower()
            aliases = str(meta.get("aliases", "") or "").lower()

            score = 0

            for term in seed_terms:
                if term in text:
                    score += 3
                if term in abstraction:
                    score += 2
                if term in header:
                    score += 1
                if term in title_en:
                    score += 4
                if term in display_name:
                    score += 4
                if term in aliases:
                    score += 2
                if term in archetype:
                    score += 2
                if term in atom_type:
                    score += 1

            if atom_type in ("clue", "event", "npc", "location"):
                score += 2
            if atom_type in ("appendix_character", "timeline"):
                score -= 1

            # prefer investigation / clue / social discovery over late climax
            if any(x in role for x in ("clue", "investigation", "information", "mystery", "social", "info_source", "clue_source", "objective", "setting", "context")):
                score += 2
            if any(x in role for x in ("combat", "climax")):
                score -= 2
            if any(x in role for x in ("endgame", "boss weakness", "plot resolution", "climax")):
                score -= 6
            if any(x in header for x in ("ritual", "banishment", "exorcism", "purification")):
                score -= 4
            if any(x in archetype for x in ("artifact", "signal", "document", "evidence", "warning", "briefing", "investigator", "captain", "control room", "ship", "vehicle", "guardian", "cult", "hazard")):
                score += 1

            score += min(len(text) // 300, 3)
            rescored.append((score, doc))

        rescored.sort(key=lambda x: x[0], reverse=True)
        mixed_docs = [doc for score, doc in rescored[:4]]

        logger.info("[SCENARIO_DB] Top rescored docs selected for synthesis: %d", len(mixed_docs))
        for i, doc in enumerate(mixed_docs, start=1):
            logger.info(
                "[SCENARIO_DB] mixed_%d meta=%s preview=%r",
                i,
                getattr(doc, "metadata", {}),
                doc.page_content[:220]
            )

        atoms = [f"ATOM {i+1}:\n{_condense_atom(doc)}" for i, doc in enumerate(mixed_docs)]
        raw_atoms_text = "\n\n".join(atoms)
        _dbg("RANDOM/CUSTOM RAW ATOMS", raw_atoms_text[:12000])
        logger.info("[SCENARIO_DB] raw_atoms_text preview: %r", raw_atoms_text[:1200])

        logger.info("Synthesizing unique scenario blueprint from atoms...")
        try:
            synth_llm = get_llm(temperature=0.2)
            synth_chain = PromptTemplate.from_template(
                read_prompt("scenario_gen.txt", prompt_dir=prompt_dir)
            ) | synth_llm

            synth_vars = {
                "themes": themes_str,
                "era_context": era_context,
                "language": lang,
                "language_name": get_language_name(lang),
                "seed": query_text,
                "atoms": raw_atoms_text,
            }

            # A) first attempt
            synth_raw = synth_chain.invoke(synth_vars)

            try:
                blueprint = extract_blueprint_json(synth_raw)

            except Exception as parse_err:
                logger.warning("Scenario blueprint parse failed: %s", parse_err)
                _dbg("SCENARIO_SYNTH RAW FAILED", str(synth_raw)[:12000])
                logger.warning("SCENARIO_SYNTH tail preview: %r", str(synth_raw)[-2000:])

                # No second LLM call here. We already attempted local JSON repair in extract_blueprint_json().
                # If parsing still fails, fall back to raw atoms instead of sending the whole broken payload back to the model.
                raise RuntimeError(f"Blueprint parse failed after local repair: {parse_err}")
                
                # except Exception as repair_err:
                #     logger.warning("Scenario blueprint repair failed: %s", repair_err)
                #     _dbg("SCENARIO_SYNTH REPAIR RAW FAILED", str(repaired_raw)[:12000])
                #     logger.warning("SCENARIO_SYNTH REPAIR tail preview: %r", str(repaired_raw)[-2000:])

                #     # B2) only if repair failed -> regenerate from atoms once
                #     logger.warning("Regenerating scenario blueprint from atoms (one retry)")
                #     regen_llm = get_llm(temperature=0.15)
                #     regen_chain = PromptTemplate.from_template(
                #         read_prompt("scenario_gen.txt", prompt_dir=prompt_dir)
                #     ) | regen_llm

                #     regen_prompt_vars = dict(synth_vars)
                #     regen_prompt_vars["atoms"] = (
                #         raw_atoms_text
                #         + "\n\nIMPORTANT:\n"
                #         + "- Return only one valid JSON object.\n"
                #         + "- Do not use markdown fences.\n"
                #         + "- Ensure all objects and arrays are properly closed.\n"
                #         + "- Do not leave trailing commas.\n"
                #     )

                #     logger.warning("SCENARIO_SYNTH: starting regenerate invoke")
                #     regen_raw = regen_chain.invoke(regen_prompt_vars)
                #     blueprint = extract_blueprint_json(regen_raw)

            logger.info(
                "[SCENARIO_SYNTH] title=%r | hook=%r | core=%r | hidden=%r",
                blueprint.get("title", ""),
                blueprint.get("inciting_hook", ""),
                blueprint.get("core_mystery", ""),
                blueprint.get("hidden_threat", ""),
            )

            scenario_atoms_text = (
                f"SCENARIO TITLE: {blueprint.get('title', 'Unknown')}\n"
                f"SETTING: {blueprint.get('era_and_setting', '')}\n"
                f"HOOK: {blueprint.get('inciting_hook', '')}\n"
                f"KEY NPC: {blueprint.get('key_npc', '')}\n"
                f"ATMOSPHERE: {blueprint.get('atmosphere_notes', '')}"
            )
            logger.info(f"Scenario blueprint synthesized: {blueprint.get('title', '?')}")

        except Exception as e:
            logger.warning(f"Scenario synthesis failed ({e}), falling back to raw atoms.")
            blueprint = {}
            scenario_atoms_text = raw_atoms_text

    # ── SHARED: save everything to DB — runs for BOTH paths ───────────────────
    _dbg("FINAL scenario_atoms_text", scenario_atoms_text)
    _dbg("FINAL blueprint (first 1000)", json.dumps(blueprint, ensure_ascii=False)[:1000])
    logger.info(f"[SESSION] Final scenario_atoms_text (first 300): {scenario_atoms_text[:300]}")
    logger.info(f"[SESSION] language being stored: {repr(lang)}")

    db.log_event("SCENARIO_GENERATED", {
        "query": query_text if req.scenarioType != 'prebuilt' else blueprint.get('title', 'prebuilt')
    })

    if isinstance(blueprint, dict) and blueprint:
        loc_id_map = {}

        # --- compact locations registry ---
        for loc in blueprint.get("locations", []):
            loc_name = str(loc.get("name", "Unknown") or "Unknown")
            tags_raw = loc.get("tags", "")
            tags_str = ", ".join(tags_raw) if isinstance(tags_raw, list) else str(tags_raw or "")

            lid = db.upsert_location(
                name=loc_name,
                description="",   # compact schema no longer stores long location prose here
                tags=tags_str
            )
            loc_id_map[loc_name] = lid

        # --- compact npc registry ---
        for npc in blueprint.get("npcs", []):
            role = str(npc.get("role", "neutral") or "neutral").lower()
            kind = "ENEMY" if role in ("enemy", "hidden_enemy") else "NPC"

            npc_name = str(npc.get("name", "Unknown") or "Unknown")
            npc_secret = str(npc.get("secret", "") or "")
            npc_motivation = str(npc.get("motivation", "") or "")

            notes_parts = []
            if npc_secret:
                notes_parts.append(f"SECRET: {npc_secret}")
            if npc_motivation:
                notes_parts.append(f"MOTIVATION: {npc_motivation}")

            aid = db.upsert_actor(
                kind=kind,
                name=npc_name,
                description=role,
                hp=10 if kind == "NPC" else 12,
                mp=0,
                san=50 if kind == "NPC" else 0,
                stats={
                    "str": 50,
                    "con": 50,
                    "dex": 50,
                    "int": 50,
                    "pow": 50,
                    "app": 40,
                    "siz": 50,
                    "edu": 50,
                },
                notes="\n".join(notes_parts).strip(),
            )

            # Safe defaults so any NPC can still function if combat/social pressure happens
            db.set_skill(aid, "Dodge", 25)
            db.set_skill(aid, "Fighting (Brawl)", 25 if kind == "NPC" else 40)
            if kind == "ENEMY":
                db.set_skill(aid, "Firearms (Handgun)", 20)

        # --- compact clues registry ---
        for clue in blueprint.get("clues", []):
            loc_name = str(clue.get("location", "") or "")
            surface = str(clue.get("content", "") or "")
            deeper = str(clue.get("true_meaning", "") or "")

            stored_content = surface
            if deeper:
                stored_content = f"{surface}\nTRUE_MEANING: {deeper}" if surface else f"TRUE_MEANING: {deeper}"

            db.upsert_clue(
                title=str(clue.get("title", "Clue") or "Clue"),
                content=stored_content,
                status="hidden",
                location_id=loc_id_map.get(loc_name)
            )

        # --- compact plot thread registry ---
        for thread in blueprint.get("plot_threads", []):
            steps_raw = thread.get("steps", 4)
            steps = int(steps_raw) if isinstance(steps_raw, (int, float, str)) else 4

            db.upsert_thread(
                name=str(thread.get("name", "Thread") or "Thread"),
                stakes=str(thread.get("stakes", "") or ""),
                max_progress=steps
            )

    scenario_setting = str(
        blueprint.get('era_and_setting') or era_context
        if isinstance(blueprint, dict)
        else era_context
    )

    first_act_no = 1
    first_scene_name = ""
    first_scene_location = ""

    if isinstance(blueprint, dict):
        acts = blueprint.get("acts") or []
        if acts:
            first_act = acts[0]
            first_act_no = int(first_act.get("act", 1) or 1)
            scenes = first_act.get("scenes") or []
            if scenes:
                first_scene_name = str(scenes[0].get("scene", "") or "")
                first_scene_location = str(scenes[0].get("location", "") or "")

    current_objective = _derive_initial_objective(
        blueprint if isinstance(blueprint, dict) else {},
        scenario_atoms_text,
        str(era_context)
    )

    cur = db.conn.cursor()
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_atoms', ?)", (str(scenario_atoms_text),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_themes', ?)", (str(themes_str),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_setting', ?)", (scenario_setting,))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('era_context', ?)", (str(era_context),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_source', ?)", (str(req.picked_seed[:100]) if req.picked_seed else "",))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('language', ?)", (lang,))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('prompt_dir', ?)", (prompt_dir,))
    cur.execute(
        "INSERT OR REPLACE INTO kv_store (key, value) VALUES ('scenario_blueprint_json', ?)",
        (json.dumps(blueprint, ensure_ascii=False) if isinstance(blueprint, dict) else "{}",)
    )
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_act', ?)", (str(first_act_no),))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_scene', ?)", (first_scene_name,))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_scene_location', ?)", (first_scene_location,))
    cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('current_objective', ?)", (current_objective,))
    db.conn.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES (?, ?)", ("scenario_era", str(era_context)))
    db.conn.commit()

    print(f"==>> [SESSION START] era_context stored    : {repr(era_context)}")
    print(f"==>> [SESSION START] scenario_setting stored: {repr(scenario_setting)}")
    print(f"==>> [SESSION START] language stored        : {repr(lang)}")

    # 4. Trigger opening narration
    # start_msg = "Start the story. Open with the first playable scene and describe the starting location."
    # return await handle_chat_logic(ChatRequest(message=start_msg, session_id=session_id))
    return await generate_opening_scene_logic(db, session_id=session_id)

def _condense_atom(doc) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    title = str(meta.get("title_en") or meta.get("display_name") or meta.get("Header_2") or "Unknown").strip()
    atom_type = str(meta.get("type") or "").strip()
    role = str(meta.get("role") or "").strip()
    archetype = str(meta.get("archetype") or "").strip()
    abstraction = str(meta.get("abstraction") or "").strip()

    text = (doc.page_content or "").strip()
    flavor = ""
    m = re.search(r"Original flavor:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        flavor = m.group(1).strip().splitlines()[0][:180]

    lines = [
        f"title: {title}",
        f"type: {atom_type}",
        f"role: {role}",
        f"archetype: {archetype}",
    ]

    if abstraction:
        lines.append(f"abstraction: {abstraction}")

    if atom_type in {"event", "clue", "location", "npc"} and flavor:
        lines.append(f"flavor_hint: {flavor}")

    return "\n".join(lines)
