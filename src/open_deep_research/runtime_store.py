"""SQLite persistence for task state, events, and successful results."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from open_deep_research.task_runtime import ResearchTask, TaskResult

from open_deep_research.evidence import Document, Evidence, EvidenceConflict, ResearchClaim


class RuntimeStore:
    """Small durable store used by the scheduler and recovery experiments."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    dependencies_json TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    timeout_seconds REAL NOT NULL,
                    retry_limit INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL,
                    result_id TEXT,
                    error TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_results (
                    task_id TEXT PRIMARY KEY,
                    result_id TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    attempts INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    retrieved_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    excerpt TEXT NOT NULL,
                    claim_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );
                CREATE TABLE IF NOT EXISTS claims (
                    claim_id TEXT PRIMARY KEY,
                    statement TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conflicts (
                    conflict_id TEXT PRIMARY KEY,
                    claim_id TEXT NOT NULL,
                    evidence_id_a TEXT NOT NULL,
                    evidence_id_b TEXT NOT NULL,
                    reason TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS research_objectives (
                    objective_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    importance REAL NOT NULL,
                    status TEXT NOT NULL,
                    linked_claim_ids_json TEXT NOT NULL,
                    search_attempts INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS adaptive_decisions (
                    decision_id TEXT PRIMARY KEY,
                    round_number INTEGER NOT NULL,
                    objective_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS report_assertions (
                    assertion_id TEXT PRIMARY KEY,
                    sentence TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    repair_action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def save_task(self, task: "ResearchTask") -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, description, dependencies_json, priority,
                    timeout_seconds, retry_limit, status, attempt_count,
                    result_id, error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status=excluded.status,
                    attempt_count=excluded.attempt_count,
                    result_id=excluded.result_id,
                    error=excluded.error,
                    updated_at=excluded.updated_at
                """,
                (
                    task.task_id,
                    task.description,
                    json.dumps(task.dependencies),
                    task.priority,
                    task.timeout_seconds,
                    task.retry_limit,
                    task.status.value,
                    task.attempt_count,
                    task.result_id,
                    task.error,
                    now,
                ),
            )

    def append_event(self, task_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO task_events (task_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?)",
                (task_id, event_type, json.dumps(payload, ensure_ascii=False), datetime.now(timezone.utc).isoformat()),
            )

    def save_result(self, task: "ResearchTask", result: "TaskResult") -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO task_results
                (task_id, result_id, content_json, attempts, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.result_id or f"result:{task.task_id}",
                    json.dumps(result.content, ensure_ascii=False, default=str),
                    result.attempts,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def commit_task_success(self, task: "ResearchTask", result: "TaskResult") -> None:
        """Atomically persist a successful task snapshot and its result.

        Recovery must never observe a committed SUCCEEDED task without the
        result that makes it resumable. Keeping both writes in one transaction
        makes the task/result pair the durable commit unit.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO tasks (
                    task_id, description, dependencies_json, priority,
                    timeout_seconds, retry_limit, status, attempt_count,
                    result_id, error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    status=excluded.status,
                    attempt_count=excluded.attempt_count,
                    result_id=excluded.result_id,
                    error=excluded.error,
                    updated_at=excluded.updated_at
                """,
                (
                    task.task_id,
                    task.description,
                    json.dumps(task.dependencies),
                    task.priority,
                    task.timeout_seconds,
                    task.retry_limit,
                    task.status.value,
                    task.attempt_count,
                    task.result_id,
                    task.error,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO task_results
                (task_id, result_id, content_json, attempts, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    task.task_id,
                    task.result_id or f"result:{task.task_id}",
                    json.dumps(result.content, ensure_ascii=False, default=str),
                    result.attempts,
                    now,
                ),
            )

    def task_snapshot(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    def result_snapshot(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM task_results WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["content"] = json.loads(result.pop("content_json"))
        return result

    def reset_tasks_for_replay(self, task_ids: set[str]) -> dict[str, int]:
        """Atomically reset tasks and invalidate their derived artifacts.

        Evidence and claims are derived from task results. Leaving them in the
        store during Replay would allow a new report to mix old and new
        evidence, so replay invalidation removes claims that reference replayed
        evidence and conflicts that reference those claims.
        """
        if not task_ids:
            return {"tasks": 0, "evidence": 0, "claims": 0, "conflicts": 0}
        placeholders = ",".join("?" for _ in task_ids)
        values = tuple(sorted(task_ids))
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            evidence_rows = connection.execute(
                f"SELECT evidence_id FROM evidence WHERE task_id IN ({placeholders})", values
            ).fetchall()
            evidence_ids = [str(row["evidence_id"]) for row in evidence_rows]
            claim_ids: list[str] = []
            conflict_count = 0
            if evidence_ids:
                evidence_placeholders = ",".join("?" for _ in evidence_ids)
                claim_rows = connection.execute(
                    "SELECT claim_id, evidence_ids_json FROM claims"
                ).fetchall()
                for row in claim_rows:
                    linked = set(json.loads(row["evidence_ids_json"]))
                    if linked.intersection(evidence_ids):
                        claim_ids.append(str(row["claim_id"]))
                if claim_ids:
                    claim_placeholders = ",".join("?" for _ in claim_ids)
                    conflict_count = int(connection.execute(
                        f"SELECT COUNT(*) FROM conflicts WHERE claim_id IN ({claim_placeholders})",
                        tuple(claim_ids),
                    ).fetchone()[0])
                    connection.execute(
                        f"DELETE FROM conflicts WHERE claim_id IN ({claim_placeholders})", tuple(claim_ids)
                    )
                    connection.execute(
                        f"DELETE FROM claims WHERE claim_id IN ({claim_placeholders})", tuple(claim_ids)
                    )
                connection.execute(
                    f"DELETE FROM evidence WHERE evidence_id IN ({evidence_placeholders})", tuple(evidence_ids)
                )
            connection.execute(
                f"UPDATE tasks SET status='PENDING', attempt_count=0, result_id=NULL, error=NULL, updated_at=? WHERE task_id IN ({placeholders})",
                (now, *values),
            )
            connection.execute(
                f"DELETE FROM task_results WHERE task_id IN ({placeholders})", values
            )
        for task_id in task_ids:
            self.append_event(task_id, "task_replay_requested", {
                "invalidated_evidence": len(evidence_ids),
                "invalidated_claims": len(claim_ids),
            })
        return {
            "tasks": len(task_ids),
            "evidence": len(evidence_ids),
            "claims": len(claim_ids),
            "conflicts": conflict_count,
        }

    def invalidate_derived_artifacts(self, task_ids: set[str]) -> dict[str, int]:
        """Remove uncommitted Evidence lineage without changing task attempts."""
        if not task_ids:
            return {"evidence": 0, "claims": 0, "conflicts": 0}
        placeholders = ",".join("?" for _ in task_ids)
        values = tuple(sorted(task_ids))
        with self._connect() as connection:
            evidence_rows = connection.execute(
                f"SELECT evidence_id FROM evidence WHERE task_id IN ({placeholders})", values
            ).fetchall()
            evidence_ids = [str(row["evidence_id"]) for row in evidence_rows]
            if not evidence_ids:
                return {"evidence": 0, "claims": 0, "conflicts": 0}
            evidence_placeholders = ",".join("?" for _ in evidence_ids)
            claim_ids = [
                str(row["claim_id"])
                for row in connection.execute(
                    "SELECT claim_id, evidence_ids_json FROM claims"
                ).fetchall()
                if set(json.loads(row["evidence_ids_json"])).intersection(evidence_ids)
            ]
            conflict_count = 0
            if claim_ids:
                claim_placeholders = ",".join("?" for _ in claim_ids)
                conflict_count = int(connection.execute(
                    f"SELECT COUNT(*) FROM conflicts WHERE claim_id IN ({claim_placeholders})",
                    tuple(claim_ids),
                ).fetchone()[0])
                connection.execute(
                    f"DELETE FROM conflicts WHERE claim_id IN ({claim_placeholders})", tuple(claim_ids)
                )
                connection.execute(
                    f"DELETE FROM claims WHERE claim_id IN ({claim_placeholders})", tuple(claim_ids)
                )
            connection.execute(
                f"DELETE FROM evidence WHERE evidence_id IN ({evidence_placeholders})", tuple(evidence_ids)
            )
        return {
            "evidence": len(evidence_ids),
            "claims": len(claim_ids),
            "conflicts": conflict_count,
        }

    def reset_task_for_replay(self, task_id: str) -> None:
        """Backward-compatible single-task replay entry point."""
        self.reset_tasks_for_replay({task_id})

    def event_count(self, task_id: str | None = None) -> int:
        with self._connect() as connection:
            if task_id is None:
                row = connection.execute("SELECT COUNT(*) AS count FROM task_events").fetchone()
            else:
                row = connection.execute("SELECT COUNT(*) AS count FROM task_events WHERE task_id = ?", (task_id,)).fetchone()
        return int(row["count"])

    def task_counts(self) -> tuple[int, int]:
        """Return total and successfully completed task counts."""
        with self._connect() as connection:
            total = connection.execute("SELECT COUNT(*) AS count FROM tasks").fetchone()["count"]
            succeeded = connection.execute(
                "SELECT COUNT(*) AS count FROM tasks WHERE status = 'SUCCEEDED'"
            ).fetchone()["count"]
        return int(total), int(succeeded)

    def save_document(self, document: Document) -> bool:
        """Insert a source document, returning false when its content is duplicate."""
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
                    (
                        document.document_id,
                        document.url,
                        document.title,
                        document.content_hash,
                        document.retrieved_at,
                    ),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def document_id_for_hash(self, content_hash: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT document_id FROM documents WHERE content_hash=?", (content_hash,)
            ).fetchone()
        return str(row["document_id"]) if row else None

    def save_evidence(self, evidence: Evidence) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO evidence VALUES (?, ?, ?, ?, ?, ?)",
                (
                    evidence.evidence_id,
                    evidence.task_id,
                    evidence.document_id,
                    evidence.excerpt,
                    evidence.claim_text,
                    evidence.confidence,
                ),
            )

    def list_evidence(self, task_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as connection:
            if task_id is None:
                rows = connection.execute(
                    """
                    SELECT e.*, d.url AS source_url, d.title AS source_title
                    FROM evidence e JOIN documents d ON d.document_id=e.document_id
                    ORDER BY e.evidence_id
                    """
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT e.*, d.url AS source_url, d.title AS source_title
                    FROM evidence e JOIN documents d ON d.document_id=e.document_id
                    WHERE e.task_id=? ORDER BY e.evidence_id
                    """,
                    (task_id,),
                ).fetchall()
        return [dict(row) for row in rows]

    def save_claim(self, claim: ResearchClaim) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO claims VALUES (?, ?, ?, ?)",
                (
                    claim.claim_id,
                    claim.statement,
                    json.dumps(claim.evidence_ids),
                    claim.status.value,
                ),
            )

    def list_claims(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM claims ORDER BY claim_id").fetchall()
        claims = []
        for row in rows:
            claim = dict(row)
            claim["evidence_ids"] = json.loads(claim.pop("evidence_ids_json"))
            claims.append(claim)
        return claims

    def save_conflict(self, conflict: EvidenceConflict) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO conflicts VALUES (?, ?, ?, ?, ?)",
                (
                    conflict.conflict_id,
                    conflict.claim_id,
                    conflict.evidence_id_a,
                    conflict.evidence_id_b,
                    conflict.reason,
                ),
            )

    def save_research_objective(self, objective: Any) -> None:
        """Persist the latest state of one adaptive research objective."""
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO research_objectives VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(objective_id) DO UPDATE SET
                    question=excluded.question,
                    importance=excluded.importance,
                    status=excluded.status,
                    linked_claim_ids_json=excluded.linked_claim_ids_json,
                    search_attempts=excluded.search_attempts,
                    updated_at=excluded.updated_at
                """,
                (
                    objective.objective_id,
                    objective.question,
                    objective.importance,
                    objective.status.value,
                    json.dumps(objective.linked_claim_ids),
                    objective.search_attempts,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def list_research_objectives(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM research_objectives ORDER BY objective_id"
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["linked_claim_ids"] = json.loads(
                item.pop("linked_claim_ids_json")
            )
            output.append(item)
        return output

    def save_adaptive_decision(
        self,
        *,
        decision_id: str,
        round_number: int,
        objective_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO adaptive_decisions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    decision_id,
                    round_number,
                    objective_id,
                    action,
                    json.dumps(payload, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def list_adaptive_decisions(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM adaptive_decisions ORDER BY round_number, decision_id"
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            output.append(item)
        return output

    def replace_report_assertions(self, assertions: list[Any]) -> None:
        """Replace the report audit atomically so resumes cannot mix report versions."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("DELETE FROM report_assertions")
            connection.executemany(
                "INSERT INTO report_assertions VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        item.assertion_id,
                        item.sentence,
                        json.dumps(item.evidence_ids),
                        item.verdict.value,
                        item.repair_action.value,
                        item.reason,
                        now,
                    )
                    for item in assertions
                ],
            )

    def list_report_assertions(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM report_assertions ORDER BY assertion_id"
            ).fetchall()
        output = []
        for row in rows:
            item = dict(row)
            item["evidence_ids"] = json.loads(item.pop("evidence_ids_json"))
            output.append(item)
        return output

    def evidence_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            return {
                table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                for table in ("documents", "evidence", "claims", "conflicts")
            }
