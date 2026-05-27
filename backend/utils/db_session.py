# backend/db/db_session.py
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _json_loads(s: Optional[str]) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


@dataclass
class SessionInfo:
    session_id: str
    db_path: str
    title: str
    setting: str


class SessionDB:
    """
    One sqlite file per session.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA synchronous = NORMAL;")
        self._migrate()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    # ----------------------------
    # Migrations
    # ----------------------------

    def _migrate(self):
        cur = self.conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS meta (
          k TEXT PRIMARY KEY,
          v TEXT
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
          id TEXT PRIMARY KEY,
          title TEXT,
          setting TEXT,
          created_at TEXT,
          updated_at TEXT
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS locations (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          description TEXT,
          parent_id TEXT,
          tags TEXT,
          state_json TEXT,
          updated_at TEXT
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS actors (
          id TEXT PRIMARY KEY,
          kind TEXT NOT NULL,           -- 'PC'|'NPC'|'ENEMY'
          name TEXT NOT NULL,
          description TEXT,
          location_id TEXT,
          hp INTEGER,
          mp INTEGER,
          san INTEGER,
          pow INTEGER,
          str INTEGER,
          con INTEGER,
          dex INTEGER,
          int INTEGER,
          app INTEGER,
          siz INTEGER,
          edu INTEGER,
          status TEXT,                  -- ok/injured/dead/insane/missing...
          notes TEXT,
          updated_at TEXT,
          FOREIGN KEY(location_id) REFERENCES locations(id)
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS actor_skills (
          actor_id TEXT NOT NULL,
          skill TEXT NOT NULL,
          value INTEGER NOT NULL,
          PRIMARY KEY (actor_id, skill),
          FOREIGN KEY(actor_id) REFERENCES actors(id) ON DELETE CASCADE
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS actor_items (
        id TEXT PRIMARY KEY,
        actor_id TEXT NOT NULL,
        item_name TEXT NOT NULL,
        quantity INTEGER NOT NULL DEFAULT 1,
        item_type TEXT,
        item_data_json TEXT,
        updated_at TEXT,
        FOREIGN KEY(actor_id) REFERENCES actors(id) ON DELETE CASCADE
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS clues (
          id TEXT PRIMARY KEY,
          title TEXT,
          content TEXT NOT NULL,
          status TEXT NOT NULL,         -- hidden/found/spent
          location_id TEXT,
          related_actor_id TEXT,
          discovered_at TEXT,
          FOREIGN KEY(location_id) REFERENCES locations(id),
          FOREIGN KEY(related_actor_id) REFERENCES actors(id)
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS kv_store (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS threads (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          progress INTEGER NOT NULL DEFAULT 0,
          max_progress INTEGER NOT NULL DEFAULT 6,
          stakes TEXT,
          updated_at TEXT
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          event_type TEXT NOT NULL,
          kind TEXT, -- kept for backward compatibility if needed
          actor_id TEXT,
          location_id TEXT,
          payload_json TEXT,
          payload TEXT, -- legacy support
          FOREIGN KEY(actor_id) REFERENCES actors(id),
          FOREIGN KEY(location_id) REFERENCES locations(id)
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS story_acts (
        id TEXT PRIMARY KEY,
        act_no INTEGER NOT NULL,
        title TEXT,
        summary TEXT,
        purpose TEXT,
        belief_shift TEXT,
        required_payoffs_json TEXT,
        module_type TEXT,
        payload_json TEXT,
        updated_at TEXT
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS story_scenes (
        id TEXT PRIMARY KEY,
        act_id TEXT NOT NULL,
        act_no INTEGER NOT NULL,
        scene_no INTEGER NOT NULL,
        name TEXT NOT NULL,
        location_id TEXT,
        scene_function TEXT,
        dramatic_question TEXT,
        entry_condition TEXT,
        exit_condition TEXT,
        trigger_text TEXT,
        description TEXT,
        what_happens TEXT,
        pressure_if_delayed TEXT,
        threat_level TEXT,
        reveals_json TEXT,
        conceals_json TEXT,
        payload_json TEXT,
        unlocked INTEGER NOT NULL DEFAULT 0,
        resolved INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT,
        FOREIGN KEY(act_id) REFERENCES story_acts(id),
        FOREIGN KEY(location_id) REFERENCES locations(id)
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS story_scene_npcs (
        scene_id TEXT NOT NULL,
        actor_id TEXT NOT NULL,
        PRIMARY KEY(scene_id, actor_id),
        FOREIGN KEY(scene_id) REFERENCES story_scenes(id) ON DELETE CASCADE,
        FOREIGN KEY(actor_id) REFERENCES actors(id) ON DELETE CASCADE
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS story_scene_clues (
        scene_id TEXT NOT NULL,
        clue_id TEXT NOT NULL,
        PRIMARY KEY(scene_id, clue_id),
        FOREIGN KEY(scene_id) REFERENCES story_scenes(id) ON DELETE CASCADE,
        FOREIGN KEY(clue_id) REFERENCES clues(id) ON DELETE CASCADE
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS story_scene_threads (
        scene_id TEXT NOT NULL,
        thread_id TEXT NOT NULL,
        PRIMARY KEY(scene_id, thread_id),
        FOREIGN KEY(scene_id) REFERENCES story_scenes(id) ON DELETE CASCADE,
        FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
        );
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS story_scene_objectives (
        id TEXT PRIMARY KEY,
        scene_id TEXT NOT NULL,
        objective_type TEXT NOT NULL,   -- trigger | dramatic_question | exit_condition | custom
        text TEXT NOT NULL,
        FOREIGN KEY(scene_id) REFERENCES story_scenes(id) ON DELETE CASCADE
        );
        """)

        # Versioning
        cur.execute("SELECT v FROM meta WHERE k='schema_version';")
        row = cur.fetchone()
        if not row:
            cur.execute("INSERT INTO meta(k,v) VALUES('schema_version','2');")
        self.conn.commit()


    # ----------------------------
    # Story graph helpers
    # ----------------------------

    def _story_scene_row_to_dict(self, row) -> Optional[Dict[str, Any]]:
        if not row:
            return None
        d = dict(row)
        d["payload"] = _json_loads(d.pop("payload_json", None)) or {}
        d["reveals"] = _json_loads(d.pop("reveals_json", None)) or []
        d["conceals"] = _json_loads(d.pop("conceals_json", None)) or []
        return d

    def clear_story_graph(self) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM story_scene_objectives")
        cur.execute("DELETE FROM story_scene_npcs")
        cur.execute("DELETE FROM story_scene_clues")
        cur.execute("DELETE FROM story_scene_threads")
        cur.execute("DELETE FROM story_scenes")
        cur.execute("DELETE FROM story_acts")
        cur.execute("DELETE FROM kv_store WHERE key IN ('current_story_scene_id','current_act','current_scene','current_scene_location','current_objective')")
        self.conn.commit()

    def upsert_story_act(
        self,
        *,
        act_no: int,
        title: str,
        summary: str = "",
        purpose: str = "",
        belief_shift: str = "",
        required_payoffs: list[str] | None = None,
        module_type: str = "",
        payload: dict | None = None,
        act_id: str | None = None,
    ) -> str:
        cur = self.conn.cursor()
        now = _utc_now_iso()
        payload_json = _json_dumps(payload or {})
        required_payoffs_json = _json_dumps(required_payoffs or [])

        target_id = act_id
        if target_id:
            cur.execute("SELECT id FROM story_acts WHERE id=?", (target_id,))
            row = cur.fetchone()
        else:
            cur.execute("SELECT id FROM story_acts WHERE act_no=?", (int(act_no),))
            row = cur.fetchone()
            target_id = row["id"] if row else uuid.uuid4().hex

        if row:
            cur.execute(
                """
                UPDATE story_acts
                SET act_no=?, title=?, summary=?, purpose=?, belief_shift=?,
                    required_payoffs_json=?, module_type=?, payload_json=?, updated_at=?
                WHERE id=?
                """,
                (
                    int(act_no), title, summary, purpose, belief_shift,
                    required_payoffs_json, module_type, payload_json, now, target_id
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO story_acts(
                    id, act_no, title, summary, purpose, belief_shift,
                    required_payoffs_json, module_type, payload_json, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    target_id, int(act_no), title, summary, purpose, belief_shift,
                    required_payoffs_json, module_type, payload_json, now
                ),
            )
        self.conn.commit()
        return target_id

    def upsert_story_scene(
        self,
        *,
        act_id: str,
        act_no: int,
        scene_no: int,
        name: str,
        location_id: str | None = None,
        payload: dict | None = None,
    ) -> str:
        cur = self.conn.cursor()
        now = _utc_now_iso()
        payload = payload or {}

        scene_function = str(payload.get("scene_function", "") or "")
        dramatic_question = str(payload.get("dramatic_question", "") or "")
        entry_condition = str(payload.get("entry_condition", "") or "")
        exit_condition = str(payload.get("exit_condition", "") or "")
        trigger_text = str(payload.get("trigger", "") or "")
        description = str(payload.get("description", "") or "")
        what_happens = str(payload.get("what_happens", "") or "")
        pressure_if_delayed = str(payload.get("pressure_if_delayed", "") or "")
        threat_level = str(payload.get("threat_level", "") or "")
        reveals_json = _json_dumps(payload.get("reveals") or [])
        conceals_json = _json_dumps(payload.get("conceals") or [])
        payload_json = _json_dumps(payload)

        cur.execute(
            "SELECT id FROM story_scenes WHERE act_id=? AND scene_no=?",
            (act_id, int(scene_no)),
        )
        row = cur.fetchone()
        scene_id = row["id"] if row else uuid.uuid4().hex

        if row:
            cur.execute(
                """
                UPDATE story_scenes
                SET act_no=?, scene_no=?, name=?, location_id=?, scene_function=?,
                    dramatic_question=?, entry_condition=?, exit_condition=?, trigger_text=?,
                    description=?, what_happens=?, pressure_if_delayed=?, threat_level=?,
                    reveals_json=?, conceals_json=?, payload_json=?, updated_at=?
                WHERE id=?
                """,
                (
                    int(act_no), int(scene_no), name, location_id, scene_function,
                    dramatic_question, entry_condition, exit_condition, trigger_text,
                    description, what_happens, pressure_if_delayed, threat_level,
                    reveals_json, conceals_json, payload_json, now, scene_id
                ),
            )
        else:
            cur.execute(
                """
                INSERT INTO story_scenes(
                    id, act_id, act_no, scene_no, name, location_id,
                    scene_function, dramatic_question, entry_condition, exit_condition,
                    trigger_text, description, what_happens, pressure_if_delayed,
                    threat_level, reveals_json, conceals_json, payload_json,
                    unlocked, resolved, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    scene_id, act_id, int(act_no), int(scene_no), name, location_id,
                    scene_function, dramatic_question, entry_condition, exit_condition,
                    trigger_text, description, what_happens, pressure_if_delayed,
                    threat_level, reveals_json, conceals_json, payload_json,
                    1 if int(act_no) == 1 and int(scene_no) == 1 else 0,
                    0,
                    now,
                ),
            )
        self.conn.commit()
        return scene_id

    def link_story_scene_npc(self, scene_id: str, actor_id: str) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO story_scene_npcs(scene_id, actor_id)
            VALUES(?, ?)
            """,
            (scene_id, actor_id),
        )
        self.conn.commit()

    def link_story_scene_clue(self, scene_id: str, clue_id: str) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO story_scene_clues(scene_id, clue_id)
            VALUES(?, ?)
            """,
            (scene_id, clue_id),
        )
        self.conn.commit()

    def link_story_scene_thread(self, scene_id: str, thread_id: str) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO story_scene_threads(scene_id, thread_id)
            VALUES(?, ?)
            """,
            (scene_id, thread_id),
        )
        self.conn.commit()

    def add_story_scene_objective(self, scene_id: str, objective_type: str, text: str) -> str:
        oid = uuid.uuid4().hex
        self.conn.execute(
            """
            INSERT INTO story_scene_objectives(id, scene_id, objective_type, text)
            VALUES(?,?,?,?)
            """,
            (oid, scene_id, objective_type, text),
        )
        self.conn.commit()
        return oid

    def set_current_story_scene(self, scene_id: str) -> None:
        """
        Set the authoritative story-scene pointer and mirror legacy kv keys.

        Newer runtime code should prefer get_current_story_scene(), but several
        prompt/context helpers still read current_act/current_scene/current_objective.
        Keeping them synchronized prevents opening/chat prompts from seeing empty
        or stale progression labels.
        """
        scene_id = str(scene_id or "").strip()
        if not scene_id:
            return

        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO kv_store(key, value) VALUES(?, ?)",
            ("current_story_scene_id", scene_id),
        )

        cur.execute(
            """
            SELECT s.act_no, s.name, l.name AS location_name
            FROM story_scenes s
            LEFT JOIN locations l ON l.id = s.location_id
            WHERE s.id=?
            """,
            (scene_id,),
        )
        row = cur.fetchone()
        if row:
            kv_pairs = {
                "current_act": str(row["act_no"] or "1"),
                "current_scene": str(row["name"] or ""),
                "current_scene_location": str(row["location_name"] or ""),
            }

            cur.execute(
                """
                SELECT text
                FROM story_scene_objectives
                WHERE scene_id=?
                ORDER BY
                  CASE objective_type
                    WHEN 'trigger' THEN 1
                    WHEN 'exit_condition' THEN 2
                    WHEN 'custom' THEN 3
                    WHEN 'dramatic_question' THEN 4
                    ELSE 5
                  END,
                  rowid
                LIMIT 1
                """,
                (scene_id,),
            )
            obj = cur.fetchone()
            if obj and str(obj["text"] or "").strip():
                kv_pairs["current_objective"] = str(obj["text"] or "").strip()

            for key, value in kv_pairs.items():
                cur.execute(
                    "INSERT OR REPLACE INTO kv_store(key, value) VALUES(?, ?)",
                    (key, value),
                )

        self.conn.commit()

    def get_current_story_scene(self) -> dict | None:
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM kv_store WHERE key='current_story_scene_id'")
        row = cur.fetchone()
        scene_id = row["value"] if row else ""

        if scene_id:
            cur.execute(
                """
                SELECT s.*, l.name AS location_name, l.description AS location_description
                FROM story_scenes s
                LEFT JOIN locations l ON l.id = s.location_id
                WHERE s.id=?
                """,
                (scene_id,),
            )
            found = cur.fetchone()
            if found:
                return self._story_scene_row_to_dict(found)

        cur.execute(
            """
            SELECT s.*, l.name AS location_name, l.description AS location_description
            FROM story_scenes s
            LEFT JOIN locations l ON l.id = s.location_id
            ORDER BY s.act_no, s.scene_no
            LIMIT 1
            """
        )
        first = cur.fetchone()
        if first:
            first_dict = self._story_scene_row_to_dict(first)
            self.set_current_story_scene(first_dict["id"])
            return first_dict
        return None

    def get_next_story_scene(self, scene_id: str) -> dict | None:
        cur = self.conn.cursor()
        cur.execute("SELECT act_no, scene_no FROM story_scenes WHERE id=?", (scene_id,))
        row = cur.fetchone()
        if not row:
            return None

        act_no = int(row["act_no"])
        scene_no = int(row["scene_no"])

        cur.execute(
            """
            SELECT s.*, l.name AS location_name, l.description AS location_description
            FROM story_scenes s
            LEFT JOIN locations l ON l.id = s.location_id
            WHERE (s.act_no > ?)
               OR (s.act_no = ? AND s.scene_no > ?)
            ORDER BY s.act_no, s.scene_no
            LIMIT 1
            """,
            (act_no, act_no, scene_no),
        )
        nxt = cur.fetchone()
        return self._story_scene_row_to_dict(nxt)

    def mark_story_scene_resolved(self, scene_id: str) -> None:
        self.conn.execute(
            "UPDATE story_scenes SET resolved=1, updated_at=? WHERE id=?",
            (_utc_now_iso(), scene_id),
        )
        self.conn.commit()

    def get_story_scene_primary_objective(self, scene_id: str) -> str:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT text
            FROM story_scene_objectives
            WHERE scene_id=?
            ORDER BY
              CASE objective_type
                WHEN 'trigger' THEN 1
                WHEN 'exit_condition' THEN 2
                WHEN 'custom' THEN 3
                WHEN 'dramatic_question' THEN 4
                ELSE 5
              END,
              rowid
            LIMIT 1
            """,
            (scene_id,),
        )
        row = cur.fetchone()
        return str(row["text"] or "").strip() if row else ""

    def list_story_scene_npcs(self, scene_id: str) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT a.*
            FROM story_scene_npcs sn
            JOIN actors a ON a.id = sn.actor_id
            WHERE sn.scene_id=?
            ORDER BY a.name
            """,
            (scene_id,),
        )
        rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["skills"] = self.get_actor_skills(d["id"])
            d["inventory"] = self.list_actor_items(d["id"])
            out.append(d)
        return out

    def list_story_scene_clues(self, scene_id: str, *, include_hidden: bool = True) -> list[dict]:
        cur = self.conn.cursor()
        if include_hidden:
            cur.execute(
                """
                SELECT c.*
                FROM story_scene_clues sc
                JOIN clues c ON c.id = sc.clue_id
                WHERE sc.scene_id=?
                ORDER BY c.title
                """,
                (scene_id,),
            )
        else:
            cur.execute(
                """
                SELECT c.*
                FROM story_scene_clues sc
                JOIN clues c ON c.id = sc.clue_id
                WHERE sc.scene_id=? AND c.status <> 'hidden'
                ORDER BY c.title
                """,
                (scene_id,),
            )
        rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["text"] = d["content"]
            out.append(d)
        return out

    def list_story_scene_threads(self, scene_id: str) -> list[dict]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT t.*
            FROM story_scene_threads st
            JOIN threads t ON t.id = st.thread_id
            WHERE st.scene_id=?
            ORDER BY t.updated_at DESC, t.name
            """,
            (scene_id,),
        )
        return [dict(r) for r in cur.fetchall()]
    # ----------------------------
    # Session lifecycle
    # ----------------------------

    def list_actor_items(self, actor_id: str) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM actor_items WHERE actor_id=? ORDER BY item_name",
            (actor_id,),
        )
        rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["item_data"] = _json_loads(d.pop("item_data_json", None)) or {}
            out.append(d)
        return out


    def add_actor_item(
        self,
        actor_id: str,
        item_name: str,
        *,
        quantity: int = 1,
        item_type: str = "",
        item_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, quantity FROM actor_items WHERE actor_id=? AND LOWER(item_name)=LOWER(?)",
            (actor_id, item_name),
        )
        row = cur.fetchone()
        now = _utc_now_iso()

        if row:
            new_qty = int(row["quantity"]) + max(1, int(quantity))
            cur.execute(
                "UPDATE actor_items SET quantity=?, item_type=?, item_data_json=?, updated_at=? WHERE id=?",
                (new_qty, item_type, _json_dumps(item_data or {}), now, row["id"]),
            )
            self.conn.commit()
            return row["id"]

        iid = uuid.uuid4().hex
        cur.execute(
            """INSERT INTO actor_items(id, actor_id, item_name, quantity, item_type, item_data_json, updated_at)
            VALUES(?,?,?,?,?,?,?)""",
            (iid, actor_id, item_name, max(1, int(quantity)), item_type, _json_dumps(item_data or {}), now),
        )
        self.conn.commit()
        return iid


    def remove_actor_item(self, actor_id: str, item_name: str, *, quantity: int = 1) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id, quantity FROM actor_items WHERE actor_id=? AND LOWER(item_name)=LOWER(?)",
            (actor_id, item_name),
        )
        row = cur.fetchone()
        if not row:
            return

        remaining = int(row["quantity"]) - max(1, int(quantity))
        if remaining > 0:
            cur.execute("UPDATE actor_items SET quantity=?, updated_at=? WHERE id=?", (remaining, _utc_now_iso(), row["id"]))
        else:
            cur.execute("DELETE FROM actor_items WHERE id=?", (row["id"],))
        self.conn.commit()

    def init_session(self, title: str, setting: str) -> SessionInfo:
        """
        Initializes the only session record in this DB if not exists.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT id,title,setting FROM sessions LIMIT 1;")
        row = cur.fetchone()
        now = _utc_now_iso()

        if row:
            # Update setting/title if changed
            sid = row["id"]
            cur.execute(
                "UPDATE sessions SET title=?, setting=?, updated_at=? WHERE id=?",
                (title or row["title"], setting or row["setting"], now, sid),
            )
            self.conn.commit()
            return SessionInfo(session_id=sid, db_path=self.db_path, title=title or row["title"], setting=setting or row["setting"])

        sid = uuid.uuid4().hex
        cur.execute(
            "INSERT INTO sessions(id,title,setting,created_at,updated_at) VALUES(?,?,?,?,?)",
            (sid, title, setting, now, now),
        )
        self.conn.commit()
        return SessionInfo(session_id=sid, db_path=self.db_path, title=title, setting=setting)

    def update_setting(self, setting: str):
        cur = self.conn.cursor()
        # Also update kv_store for compatibility
        cur.execute("INSERT OR REPLACE INTO kv_store (key, value) VALUES ('setting', ?)", (setting,))
        
        cur.execute("SELECT id FROM sessions LIMIT 1;")
        row = cur.fetchone()
        if not row:
            return
        cur.execute("UPDATE sessions SET setting=?, updated_at=? WHERE id=?", (setting, _utc_now_iso(), row["id"]))
        self.conn.commit()

    # ----------------------------
    # Locations
    # ----------------------------

    def upsert_location(
        self,
        name: str,
        *,
        description: str = "",
        parent_id: Optional[str] = None,
        tags: str = "",
        state: Optional[Dict[str, Any]] = None,
        location_id: Optional[str] = None,
    ) -> str:
        """Create or update a location by id, otherwise by case-insensitive name."""
        name = (name or "").strip() or "Unknown Location"
        now = _utc_now_iso()
        state_json = _json_dumps(state) if state is not None else None

        cur = self.conn.cursor()
        lid = location_id

        row = None
        if lid:
            cur.execute("SELECT id FROM locations WHERE id=?", (lid,))
            row = cur.fetchone()
        if not row:
            cur.execute("SELECT id FROM locations WHERE LOWER(name)=LOWER(?)", (name,))
            row = cur.fetchone()
            if row:
                lid = row["id"]

        if row:
            cur.execute(
                """UPDATE locations
                   SET name=?,
                       description=COALESCE(NULLIF(?, ''), description),
                       parent_id=COALESCE(?, parent_id),
                       tags=COALESCE(NULLIF(?, ''), tags),
                       state_json=COALESCE(?, state_json),
                       updated_at=?
                   WHERE id=?""",
                (name, description, parent_id, tags, state_json, now, lid),
            )
        else:
            lid = lid or uuid.uuid4().hex
            cur.execute(
                """INSERT INTO locations(id,name,description,parent_id,tags,state_json,updated_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (lid, name, description, parent_id, tags, state_json, now),
            )
        self.conn.commit()
        return str(lid)

    def get_location(self, location_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM locations WHERE id=?", (location_id,))
        r = cur.fetchone()
        if not r:
            return None
        d = dict(r)
        d["state"] = _json_loads(d.pop("state_json", None)) or {}
        return d

    def get_location_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM locations WHERE LOWER(name)=LOWER(?)", (name,))
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["state"] = _json_loads(d.pop("state_json", None)) or {}
        return d
    
    # ----------------------------
    # Actors (PC/NPC/ENEMY)
    # ----------------------------

    def upsert_actor(
        self,
        kind: str,
        name: str,
        *,
        description: str = "",
        location_id: Optional[str] = None,
        hp: Optional[int] = None,
        mp: Optional[int] = None,
        san: Optional[int] = None,
        stats: Optional[Dict[str, int]] = None,
        status: str = "ok",
        notes: str = "",
        actor_id: Optional[str] = None,
    ) -> str:
        now = _utc_now_iso()
        stats = stats or {}

        cur = self.conn.cursor()
        aid = actor_id
        row = None
        if aid:
            cur.execute("SELECT id FROM actors WHERE id=?", (aid,))
            row = cur.fetchone()
        if not row:
            cur.execute(
                "SELECT id FROM actors WHERE kind=? AND LOWER(name)=LOWER(?)",
                (kind, name),
            )
            row = cur.fetchone()
            if row:
                aid = row["id"]
        if not aid:
            aid = uuid.uuid4().hex
        exists = row is not None

        vals = dict(
            id=aid, kind=kind, name=name, description=description, location_id=location_id,
            hp=hp, mp=mp, san=san,
            pow=stats.get("pow"), str=stats.get("str"), con=stats.get("con"), dex=stats.get("dex"),
            int=stats.get("int"), app=stats.get("app"), siz=stats.get("siz"), edu=stats.get("edu"),
            status=status, notes=notes, updated_at=now
        )

        if exists:
            cur.execute(
                """UPDATE actors SET
                    kind=:kind, name=:name, description=:description, location_id=:location_id,
                    hp=:hp, mp=:mp, san=:san,
                    pow=:pow, str=:str, con=:con, dex=:dex, int=:int, app=:app, siz=:siz, edu=:edu,
                    status=:status, notes=:notes, updated_at=:updated_at
                   WHERE id=:id""",
                vals,
            )
        else:
            cur.execute(
                """INSERT INTO actors(
                    id,kind,name,description,location_id,hp,mp,san,
                    pow,str,con,dex,int,app,siz,edu,
                    status,notes,updated_at
                   ) VALUES(
                    :id,:kind,:name,:description,:location_id,:hp,:mp,:san,
                    :pow,:str,:con,:dex,:int,:app,:siz,:edu,
                    :status,:notes,:updated_at
                   )""",
                vals,
            )
        self.conn.commit()
        return aid

    def patch_actor(
        self,
        actor_id: str,
        *,
        kind: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        location_id: Optional[str] = None,
        hp: Optional[int] = None,
        mp: Optional[int] = None,
        san: Optional[int] = None,
        stats: Optional[Dict[str, int]] = None,
        status: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Partial actor update: only writes fields that are explicitly provided.
        Prevents accidental NULL-wiping of HP/SAN/MP/stats during location changes.
        """
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM actors WHERE id=?", (actor_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Actor not found: {actor_id}")

        current = dict(row)
        stats = stats or {}

        vals = {
            "id": actor_id,
            "kind": kind if kind is not None else current["kind"],
            "name": name if name is not None else current["name"],
            "description": description if description is not None else current["description"],
            "location_id": location_id if location_id is not None else current["location_id"],
            "hp": hp if hp is not None else current["hp"],
            "mp": mp if mp is not None else current["mp"],
            "san": san if san is not None else current["san"],
            "pow": stats.get("pow", current["pow"]),
            "str": stats.get("str", current["str"]),
            "con": stats.get("con", current["con"]),
            "dex": stats.get("dex", current["dex"]),
            "int": stats.get("int", current["int"]),
            "app": stats.get("app", current["app"]),
            "siz": stats.get("siz", current["siz"]),
            "edu": stats.get("edu", current["edu"]),
            "status": status if status is not None else current["status"],
            "notes": notes if notes is not None else current["notes"],
            "updated_at": _utc_now_iso(),
        }

        cur.execute(
            """UPDATE actors SET
                kind=:kind, name=:name, description=:description, location_id=:location_id,
                hp=:hp, mp=:mp, san=:san,
                pow=:pow, str=:str, con=:con, dex=:dex, int=:int, app=:app, siz=:siz, edu=:edu,
                status=:status, notes=:notes, updated_at=:updated_at
            WHERE id=:id""",
            vals,
        )
        self.conn.commit()
        return actor_id

    def set_skill(self, actor_id: str, skill: str, value: int):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO actor_skills(actor_id,skill,value) VALUES(?,?,?) "
            "ON CONFLICT(actor_id,skill) DO UPDATE SET value=excluded.value",
            (actor_id, skill, int(value)),
        )
        self.conn.commit()

    def get_actor_skills(self, actor_id: str) -> Dict[str, int]:
        cur = self.conn.cursor()
        cur.execute("SELECT skill,value FROM actor_skills WHERE actor_id=? ORDER BY skill", (actor_id,))
        return {r["skill"]: int(r["value"]) for r in cur.fetchall()}

    def list_actors(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        if kind:
            cur.execute("SELECT * FROM actors WHERE kind=? ORDER BY name", (kind,))
        else:
            cur.execute("SELECT * FROM actors ORDER BY kind,name")
        rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            skills = self.get_actor_skills(d["id"])
            d["skills"] = skills # Return as dict directly
            # Also populate stats dict for compatibility
            d["stats"] = {
                "STR": d["str"], "CON": d["con"], "SIZ": d["siz"],
                "DEX": d["dex"], "APP": d["app"], "INT": d["int"],
                "POW": d["pow"], "EDU": d["edu"]
            }
            d["inventory"] = self.list_actor_items(d["id"])

            out.append(d)
        return out

    # ----------------------------
    # Clues & threads
    # ----------------------------

    def add_clue(
        self,
        content: str = "",
        text: str = "", # Alias for content
        title: str = "Clue",
        *,
        status: str = "hidden",
        location_id: Optional[str] = None,
        related_actor_id: Optional[str] = None,
    ) -> str:
        # Compatibility fix: Engine uses 'text', DB uses 'content'
        final_content = text if text else content
        
        cid = uuid.uuid4().hex
        cur = self.conn.cursor()
        cur.execute(
            """INSERT INTO clues(id,title,content,status,location_id,related_actor_id,discovered_at)
               VALUES(?,?,?,?,?,?,NULL)""",
            (cid, title, final_content, status, location_id, related_actor_id),
        )
        self.conn.commit()
        return cid

    def upsert_clue(
        self,
        title: str = "Clue",
        content: str = "",
        status: str = "hidden",
        location_id: Optional[str] = None,
        related_actor_id: Optional[str] = None,
        clue_id: Optional[str] = None,
    ) -> str:
        cur = self.conn.cursor()
        now = _utc_now_iso()
        if clue_id:
            cur.execute("SELECT id FROM clues WHERE id=?", (clue_id,))
            if cur.fetchone():
                cur.execute(
                    "UPDATE clues SET title=?, content=?, status=?, location_id=?, discovered_at=? WHERE id=?",
                    (title, content, status, location_id, now if status == "found" else None, clue_id),
                )
                self.conn.commit()
                return clue_id
        # Check by title to avoid duplicates
        cur.execute("SELECT id FROM clues WHERE LOWER(title)=LOWER(?)", (title,))
        row = cur.fetchone()
        if row:
            cid = row["id"]
            cur.execute(
                "UPDATE clues SET content=?, status=?, location_id=?, discovered_at=? WHERE id=?",
                (content, status, location_id, now if status == "found" else None, cid),
            )
            self.conn.commit()
            return cid
        # Insert new
        cid = uuid.uuid4().hex
        cur.execute(
            "INSERT INTO clues(id,title,content,status,location_id,related_actor_id,discovered_at) VALUES(?,?,?,?,?,?,?)",
            (cid, title, content, status, location_id, related_actor_id, now if status == "found" else None),
        )
        self.conn.commit()
        return cid
    
    def list_clues(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        if status:
            cur.execute("SELECT * FROM clues WHERE status=? ORDER BY discovered_at DESC NULLS LAST LIMIT ?", (status, limit))
        else:
            cur.execute("SELECT * FROM clues ORDER BY discovered_at DESC NULLS LAST LIMIT ?", (limit,))
        
        # Map content to text for compatibility
        res = []
        for r in cur.fetchall():
            d = dict(r)
            d['text'] = d['content']
            res.append(d)
        return res

    def upsert_thread(
        self,
        name: str,
        *,
        progress: int = 0,
        max_progress: int = 6,
        stakes: str = "",
        thread_id: Optional[str] = None,
    ) -> str:
        name = (name or "").strip() or "Story Thread"
        now = _utc_now_iso()
        cur = self.conn.cursor()

        tid = thread_id
        row = None
        if tid:
            cur.execute("SELECT id FROM threads WHERE id=?", (tid,))
            row = cur.fetchone()
        if not row:
            cur.execute("SELECT id FROM threads WHERE LOWER(name)=LOWER(?)", (name,))
            row = cur.fetchone()
            if row:
                tid = row["id"]
        if not tid:
            tid = uuid.uuid4().hex

        if row:
            cur.execute(
                "UPDATE threads SET name=?, progress=?, max_progress=?, stakes=?, updated_at=? WHERE id=?",
                (name, int(progress), int(max_progress), stakes, now, tid),
            )
        else:
            cur.execute(
                "INSERT INTO threads(id,name,progress,max_progress,stakes,updated_at) VALUES(?,?,?,?,?,?)",
                (tid, name, int(progress), int(max_progress), stakes, now),
            )
        self.conn.commit()
        return tid

    def list_threads(self) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM threads ORDER BY updated_at DESC")
        return [dict(r) for r in cur.fetchall()]

    # ----------------------------
    # Events (source of truth)
    # ----------------------------

    def log_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None, *, actor_id: Optional[str] = None, location_id: Optional[str] = None):
        """
        Logs an event. Supports both (kind, payload) and kwargs.
        """
        final_payload = payload if payload is not None else {}
        
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO events(ts,event_type,kind,actor_id,location_id,payload_json) VALUES(?,?,?,?,?,?)",
            (_utc_now_iso(), event_type, event_type, actor_id, location_id, _json_dumps(final_payload)),
        )
        self.conn.commit()

    def list_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            payload = _json_loads(d.pop("payload_json", None))
            if payload is None:
                payload = _json_loads(d.get("payload"))
            d["payload"] = payload or {}
            out.append(d)
        return list(reversed(out))

    # ----------------------------
    # Prompt context pack
    # ----------------------------

    def build_prompt_state_pack(self, *, limit_events: int = 10) -> Dict[str, str]:
        pcs = self.list_actors("PC")
        npcs = self.list_actors("NPC")

        def actor_line(a: Dict[str, Any], *, include_all_skills: bool = False) -> str:
            skills = a.get("skills") or {}
            if include_all_skills:
                skill_items = sorted(skills.items(), key=lambda kv: kv[0].lower())
            else:
                skill_items = [(k, v) for k, v in skills.items() if v > 20]
                skill_items = sorted(skill_items, key=lambda kv: kv[1], reverse=True)

            skills_s = ", ".join([f"{k} {v}" for k, v in skill_items]) if skill_items else ""

            stats = []
            for k in ["str", "con", "dex", "int", "app", "pow", "siz", "edu"]:
                if a.get(k) is not None:
                    stats.append(f"{k.upper()} {a.get(k)}")
            stats_s = ", ".join(stats)

            items = a.get("inventory") or []
            items_s = ", ".join(
                [f"{it['item_name']} x{it.get('quantity',1)}" if int(it.get("quantity",1)) > 1 else it["item_name"] for it in items]
            ) if items else ""

            parts = [a["name"]]
            if a.get("san") is not None:
                parts.append(f"SAN {a['san']}")
            if a.get("hp") is not None:
                parts.append(f"HP {a['hp']}")
            if stats_s:
                parts.append(stats_s)
            if skills_s:
                parts.append(f"Skills: {skills_s}")
            if items_s:
                parts.append(f"Inventory: {items_s}")
            return " — ".join(parts)
        
        investigators_text = "\n".join([f"- {actor_line(a, include_all_skills=True)}" for a in pcs]) if pcs else "(none)"
        npcs_text = "\n".join([f"- {actor_line(a)}" for a in npcs[:8]]) if npcs else "(none)"

        loc_id = None
        if pcs:
            loc_id = pcs[0].get("location_id")
        location_text = "(unknown)"
        if loc_id:
            loc = self.get_location(loc_id)
            if loc:
                st = loc.get("state", {})
                state_bits = ", ".join([f"{k}={v}" for k,v in list(st.items())[:10]]) if isinstance(st, dict) else ""
                location_text = f"{loc['name']}. {loc.get('description','')}".strip()
                if state_bits:
                    location_text += f"\nState: {state_bits}"

        threads = self.list_threads()
        threads_text = "\n".join([f"- {t['name']}: {t['progress']}/{t['max_progress']} ({t.get('stakes','')})" for t in threads[:6]]) if threads else "(none)"

        found = self.list_clues(status="found", limit=8)
        clues_text = "\n".join([f"- {c['title']}: {c['content'][:160]}" for c in found]) if found else "(none)"

        events = self.list_events(limit=limit_events)
        def ev_line(e):
            payload = e.get("payload") or {}
            brief = payload.get("brief") or payload.get("note") or ""
            if brief:
                brief = str(brief)[:140]
            return f"- {e['ts']} {e['event_type']}" + (f": {brief}" if brief else "")
        recent_events_text = "\n".join([ev_line(e) for e in events]) if events else "(none)"

        return dict(
            investigators_text=investigators_text,
            npcs_text=npcs_text,
            location_text=location_text,
            threads_text=threads_text,
            clues_text=clues_text,
            recent_events_text=recent_events_text,
        )


def create_session_db_file(base_dir: str, title: str, setting: str) -> SessionInfo:
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    sid = uuid.uuid4().hex
    db_path = str(Path(base_dir) / f"session_{sid}.sqlite")

    db = SessionDB(db_path)
    info = db.init_session(title=title, setting=setting)
    db.close()
    return info