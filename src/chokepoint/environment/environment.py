"""Per-scenario execution environment.

Replaces the module-level ``MOCK_STATE`` singleton used by the pre-audit
harness. Each scenario owns an isolated ``Environment`` built from its own
declared state, so a scenario can reference whatever files, mailboxes, and
tables its prompt actually talks about.

The singleton was the root cause of two audit defects: benign tasks referenced
resources that did not exist (inflating the false-rejection rate), and mutations
from one scenario leaked into the next.
"""

from __future__ import annotations

import re
import sqlite3
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


class InjectionError(RuntimeError):
    """Raised when a scenario's payload cannot be placed into the environment.

    This is deliberately fatal. The pre-audit harness silently skipped
    unresolvable injections, so 38 of 43 attack scenarios ran with no payload
    present while still counting as attack trials.
    """


@dataclass
class ToolCall:
    """One attempted tool invocation, whether or not it executed."""

    tool: str
    args: dict[str, Any]
    result: str
    blocked_by: str | None = None

    @property
    def executed(self) -> bool:
        return self.blocked_by is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "args": self.args,
            "result": self.result,
            "blocked_by": self.blocked_by,
            "executed": self.executed,
        }


# Locator grammar:
#   email:<email_id>
#   file:<path>
#   sql:<table>.<column>[#<row_index>]     (row_index defaults to 0)
_LOCATOR_RE = re.compile(
    r"^(?P<channel>email|file|sql):(?P<target>.+)$",
    re.DOTALL,
)
_SQL_TARGET_RE = re.compile(r"^(?P<table>[^.]+)\.(?P<column>[^#]+)(?:#(?P<row>\d+))?$")


@dataclass
class Environment:
    """Isolated mailbox, filesystem, and database for a single scenario run."""

    emails: list[dict[str, Any]] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    db: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    log: list[ToolCall] = field(default_factory=list)

    @classmethod
    def from_spec(cls, spec: dict[str, Any] | None) -> Environment:
        """Build an environment from a scenario's ``environment`` block."""
        spec = spec or {}
        return cls(
            emails=deepcopy(spec.get("emails", [])),
            files=dict(spec.get("files", {})),
            db=deepcopy(spec.get("db", {})),
        )

    # ── payload placement ────────────────────────────────────────────────────

    def inject(self, locator: str, payload: str, mode: str = "append") -> None:
        """Place ``payload`` at ``locator``, raising if the target is absent.

        Args:
            locator: A ``channel:target`` string (see the locator grammar).
            payload: The adversarial text to place.
            mode: ``append`` to concatenate onto existing content, ``replace``
                to overwrite it.

        Raises:
            InjectionError: If the locator is malformed or names a resource
                that does not exist in this environment.
        """
        match = _LOCATOR_RE.match(locator)
        if not match:
            raise InjectionError(
                f"Malformed locator {locator!r}. Expected 'email:<id>', "
                f"'file:<path>', or 'sql:<table>.<column>[#<row>]'."
            )

        channel = match.group("channel")
        target = match.group("target")

        if channel == "email":
            self._inject_email(target, payload, mode)
        elif channel == "file":
            self._inject_file(target, payload, mode)
        else:
            self._inject_sql(target, payload, mode)

    def _combine(self, existing: str, payload: str, mode: str) -> str:
        if mode == "replace":
            return payload
        if mode == "append":
            return f"{existing}\n{payload}" if existing else payload
        raise InjectionError(f"Unknown injection mode {mode!r}; use 'append' or 'replace'.")

    def _inject_email(self, email_id: str, payload: str, mode: str) -> None:
        for email in self.emails:
            if str(email.get("id")) == email_id:
                email["body"] = self._combine(email.get("body", ""), payload, mode)
                return
        available = [str(e.get("id")) for e in self.emails]
        raise InjectionError(
            f"No email with id {email_id!r} in this environment. Available ids: {available}"
        )

    def _inject_file(self, path: str, payload: str, mode: str) -> None:
        if path not in self.files:
            raise InjectionError(
                f"No file {path!r} in this environment. Available files: "
                f"{sorted(self.files)}"
            )
        self.files[path] = self._combine(self.files[path], payload, mode)

    def _inject_sql(self, target: str, payload: str, mode: str) -> None:
        match = _SQL_TARGET_RE.match(target)
        if not match:
            raise InjectionError(
                f"Malformed SQL locator target {target!r}. Expected "
                f"'<table>.<column>' or '<table>.<column>#<row_index>'."
            )
        table, column = match.group("table"), match.group("column")
        row_index = int(match.group("row") or 0)

        if table not in self.db:
            raise InjectionError(
                f"No table {table!r} in this environment. Available tables: {sorted(self.db)}"
            )
        rows = self.db[table]
        if row_index >= len(rows):
            raise InjectionError(
                f"Table {table!r} has {len(rows)} row(s); cannot inject at index {row_index}."
            )
        if column not in rows[row_index]:
            raise InjectionError(
                f"No column {column!r} in table {table!r}. Available columns: "
                f"{sorted(rows[row_index])}"
            )
        rows[row_index][column] = self._combine(
            str(rows[row_index][column]), payload, mode
        )

    # ── execution logging ────────────────────────────────────────────────────

    def record(
        self,
        tool: str,
        args: dict[str, Any],
        result: str,
        blocked_by: str | None = None,
    ) -> None:
        """Append a tool invocation to this run's execution log."""
        self.log.append(ToolCall(tool=tool, args=args, result=result, blocked_by=blocked_by))

    @property
    def executed_calls(self) -> list[ToolCall]:
        """Calls that actually reached the tool implementation."""
        return [c for c in self.log if c.executed]

    # ── SQL backing store ────────────────────────────────────────────────────

    def open_sqlite(self) -> sqlite3.Connection:
        """Materialize ``db`` into an in-memory SQLite database.

        Using real SQLite rather than substring-matching on the query means
        destructive statements have real, observable effects and read queries
        return real result sets, so an exfiltration attempt is distinguishable
        from a legitimate lookup by what it actually returned.
        """
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        for table, rows in self.db.items():
            if not rows:
                continue
            columns = list(rows[0].keys())
            col_defs = ", ".join(f'"{c}"' for c in columns)
            conn.execute(f'CREATE TABLE "{table}" ({col_defs})')
            placeholders = ", ".join("?" for _ in columns)
            conn.executemany(
                f'INSERT INTO "{table}" VALUES ({placeholders})',
                [tuple(row.get(c) for c in columns) for row in rows],
            )
        conn.commit()
        return conn
