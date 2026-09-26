"""Agent tool suite, bound to a per-scenario :class:`Environment`.

The pre-audit tools closed over a module-level dictionary, which made isolated
per-scenario state impossible. Here ``build_tools(env)`` returns a fresh tool
list whose implementations read and write only ``env``.
"""

from __future__ import annotations

import json
import sqlite3

from langchain_core.tools import StructuredTool

from chokepoint.environment.environment import Environment

TOOL_NAMES = ["read_email", "send_email", "read_file", "write_file", "execute_sql_query"]


def build_tools(env: Environment) -> list[StructuredTool]:
    """Construct the five-tool agent suite operating on ``env``."""

    def read_email(email_id: str | None = None, search_query: str | None = None) -> str:
        """Read an email by ID, or list emails matching a search query.
        If no arguments are provided, lists all emails."""
        if email_id is not None:
            for email in env.emails:
                if str(email.get("id")) == str(email_id):
                    result = json.dumps(email)
                    env.record("read_email", {"email_id": email_id}, result)
                    return result
            result = f"Error: no email with id {email_id}."
            env.record("read_email", {"email_id": email_id}, result)
            return result

        if search_query:
            needle = search_query.lower()
            matches = [
                e
                for e in env.emails
                if needle in str(e.get("subject", "")).lower()
                or needle in str(e.get("body", "")).lower()
            ]
            result = json.dumps(matches)
            env.record("read_email", {"search_query": search_query}, result)
            return result

        result = json.dumps(
            [
                {"id": e.get("id"), "subject": e.get("subject"), "sender": e.get("sender")}
                for e in env.emails
            ]
        )
        env.record("read_email", {}, result)
        return result

    def send_email(to_address: str, subject: str, body: str) -> str:
        """Send an email to a specified address."""
        result = f"Email sent successfully to {to_address}."
        env.record(
            "send_email",
            {"to_address": to_address, "subject": subject, "body": body},
            result,
        )
        return result

    def read_file(file_path: str) -> str:
        """Read the contents of a file from the file system."""
        content = env.files.get(file_path)
        if content is None:
            content = f"Error: file not found: {file_path}"
        env.record("read_file", {"file_path": file_path}, content)
        return content

    def write_file(file_path: str, content: str) -> str:
        """Write contents to a file in the file system."""
        env.files[file_path] = content
        result = f"Successfully wrote {len(content)} bytes to {file_path}."
        env.record("write_file", {"file_path": file_path, "content": content}, result)
        return result

    def execute_sql_query(query: str) -> str:
        """Execute a SQL query against the database."""
        conn = env.open_sqlite()
        try:
            cursor = conn.execute(query)
            if cursor.description is not None:
                rows = [dict(r) for r in cursor.fetchall()]
                result = json.dumps(rows, default=str)
            else:
                conn.commit()
                result = f"Query OK. {cursor.rowcount} row(s) affected."
                _sync_back(env, conn)
        except sqlite3.Error as exc:
            result = f"SQL error: {exc}"
        finally:
            conn.close()

        env.record("execute_sql_query", {"query": query}, result)
        return result

    specs = [
        (read_email, "read_email"),
        (send_email, "send_email"),
        (read_file, "read_file"),
        (write_file, "write_file"),
        (execute_sql_query, "execute_sql_query"),
    ]
    return [
        StructuredTool.from_function(func=fn, name=name, description=fn.__doc__ or name)
        for fn, name in specs
    ]


def _sync_back(env: Environment, conn: sqlite3.Connection) -> None:
    """Persist post-mutation table contents back into the environment.

    Destructive statements must leave observable damage, otherwise a scenario
    cannot distinguish an attack that ran from one that was blocked.
    """
    for table in list(env.db):
        try:
            rows = conn.execute(f'SELECT * FROM "{table}"').fetchall()
            env.db[table] = [dict(r) for r in rows]
        except sqlite3.Error:
            # Table was dropped by the executed statement.
            env.db[table] = []


def tool_names() -> list[str]:
    """Names of every tool in the suite, for scenario validation."""
    return list(TOOL_NAMES)
