"""Regression tests for audit defect D3: shared, scenario-blind environment.

The pre-audit harness used one module-level ``MOCK_STATE`` for every scenario.
It held 2 emails, 2 files, and 1 table, while all 52 benign scenarios referenced
resources absent from it, so tool calls errored and the failure was scored as a
false rejection. Mutations also leaked between scenarios.
"""

from __future__ import annotations

import json

from chokepoint.environment.environment import Environment
from chokepoint.tools.registry import build_tools


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


class TestIsolation:
    def test_each_environment_is_independent(self):
        spec = {"files": {"a.txt": "original"}}
        env_a = Environment.from_spec(spec)
        env_b = Environment.from_spec(spec)

        _tool(build_tools(env_a), "write_file").invoke(
            {"file_path": "a.txt", "content": "mutated"}
        )

        assert env_a.files["a.txt"] == "mutated"
        assert env_b.files["a.txt"] == "original"

    def test_db_mutations_do_not_leak_across_environments(self):
        spec = {"db": {"t": [{"id": 1}]}}
        env_a = Environment.from_spec(spec)
        env_b = Environment.from_spec(spec)
        _tool(build_tools(env_a), "execute_sql_query").invoke({"query": "DELETE FROM t"})
        assert env_a.db["t"] == []
        assert env_b.db["t"] == [{"id": 1}]

    def test_execution_log_starts_empty(self):
        env = Environment.from_spec({"files": {}})
        assert env.log == []


class TestScenarioDeclaredResourcesExist:
    def test_declared_file_is_readable(self):
        env = Environment.from_spec({"files": {"finance/close.md": "content"}})
        result = _tool(build_tools(env), "read_file").invoke({"file_path": "finance/close.md"})
        assert result == "content"

    def test_undeclared_file_returns_an_error_string(self):
        env = Environment.from_spec({"files": {}})
        result = _tool(build_tools(env), "read_file").invoke({"file_path": "ghost.md"})
        assert "not found" in result


class TestRealSqlSemantics:
    def test_select_returns_real_rows(self):
        env = Environment.from_spec(
            {"db": {"inventory": [{"sku": "A", "qty": 4}, {"sku": "B", "qty": 40}]}}
        )
        result = _tool(build_tools(env), "execute_sql_query").invoke(
            {"query": "SELECT sku FROM inventory WHERE qty < 10"}
        )
        assert json.loads(result) == [{"sku": "A"}]

    def test_drop_table_actually_destroys_data(self):
        """Destructive statements must leave observable damage."""
        env = Environment.from_spec({"db": {"vendors": [{"id": 1}]}})
        _tool(build_tools(env), "execute_sql_query").invoke({"query": "DROP TABLE vendors"})
        assert env.db["vendors"] == []

    def test_invalid_sql_reports_an_error_rather_than_succeeding(self):
        env = Environment.from_spec({"db": {"t": [{"id": 1}]}})
        result = _tool(build_tools(env), "execute_sql_query").invoke({"query": "NOT SQL"})
        assert "SQL error" in result


class TestExecutionLogging:
    def test_every_call_is_recorded_with_arguments(self):
        env = Environment.from_spec({"files": {"a.txt": "x"}})
        _tool(build_tools(env), "read_file").invoke({"file_path": "a.txt"})
        assert len(env.log) == 1
        assert env.log[0].tool == "read_file"
        assert env.log[0].args == {"file_path": "a.txt"}
        assert env.log[0].executed is True

    def test_blocked_calls_are_distinguishable_from_executed_ones(self):
        env = Environment.from_spec({})
        env.record("send_email", {"to_address": "e@x"}, "[BLOCKED]", blocked_by="llm_judge")
        assert env.log[0].executed is False
        assert env.executed_calls == []
