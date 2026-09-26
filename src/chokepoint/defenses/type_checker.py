"""
type_checker.py — Strict parameter type-checking and sanitization defense wrapper.

Implements OWASP-informed input validation for LLM tool use.  Every argument
that an agent passes to a wrapped tool is run through a chain of configurable
validation rules *before* the tool is invoked.  If any rule fires, the call is
blocked, a structured security log entry is recorded, and the tool returns a
safe error string instead of executing.

Key design decisions
--------------------
* **Configurable rule bank** – all patterns live in a plain dict (or YAML file)
  so researchers can add / remove / tune rules without touching code.
* **Composable validators** – each validation category (path, SQL, email,
  string) is a standalone classmethod that returns a ``ValidationResult``.
* **Zero false-negative bias** – the wrapper intentionally errs on the
  side of blocking.  Researchers can relax rules via the config.

References
----------
* OWASP Input Validation Cheat Sheet
* OWASP SQL Injection Prevention Cheat Sheet
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from copy import deepcopy
from enum import Enum
from typing import Any

import yaml
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("chokepoint.defenses.type_checker")
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Validation result enum
# ---------------------------------------------------------------------------


class ValidationResult(str, Enum):
    """Outcome of a single validation check.

    Using ``str`` as a mixin so the value serialises cleanly in JSON logs.
    """

    PASS = "PASS"
    FAIL_PATH_TRAVERSAL = "FAIL_PATH_TRAVERSAL"
    FAIL_DESTRUCTIVE_SQL = "FAIL_DESTRUCTIVE_SQL"
    FAIL_SUSPICIOUS_EMAIL = "FAIL_SUSPICIOUS_EMAIL"
    FAIL_INJECTION_MARKER = "FAIL_INJECTION_MARKER"


# ---------------------------------------------------------------------------
# Security-log entry model
# ---------------------------------------------------------------------------


class SecurityLogEntry(BaseModel):
    """Structured record emitted every time a call is blocked.

    Attributes:
        tool_name: Name of the LangChain tool whose call was blocked.
        args: Dictionary of arguments the agent attempted to pass.
        rule_violated: The ``ValidationResult`` variant that fired.
        matched_pattern: The exact regex / literal that matched the input.
    """

    tool_name: str
    args: dict[str, Any]
    rule_violated: ValidationResult
    matched_pattern: str


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    "file_path": {
        "traversal_patterns": [
            r"\.\./",          # Unix-style traversal
            r"\.\\.\\",        # Windows-style traversal (escaped backslash)
            r"\.\.\\",         # Bare backslash variant
        ],
        "sensitive_paths": [
            r"/etc/passwd",
            r"/etc/shadow",
            r"/etc/hosts",
            r"/proc/self",
            r"C:\\Windows\\System32",
            r"C:/Windows/System32",
        ],
        "shell_metacharacters": [
            r";",
            r"\|",
            r"&",
            r"\$",
            r"`",
        ],
    },
    "sql": {
        "destructive_keywords": [
            r"\bDROP\b",
            r"\bDELETE\b",
            r"\bTRUNCATE\b",
            r"\bALTER\b",
            r"\bINSERT\s+INTO\s+\S+\s+SELECT\b",
            r"\bUNION\b",
        ],
        "allowed_statements": [
            r"^\s*SELECT\b",
        ],
    },
    "email": {
        "valid_format": r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        "suspicious_domains": [
            r"@evil\.com$",
            r"@malicious\.",
            r"@tempmail\.",
            r"@throwaway\.",
            r"@guerrillamail\.",
            r"@sharklasers\.",
            r"@yopmail\.",
            r"@mailinator\.",
            r"@10minutemail\.",
        ],
    },
    "string": {
        "injection_markers": [
            r"(?i)ignore\s+(all\s+)?previous\s+instructions",
            r"(?i)ignore\s+previous",
            r"(?i)system\s*:",
            r"(?i)ADMIN\s+OVERRIDE",
            r"(?i)you\s+are\s+now\s+(in\s+)?developer\s+mode",
            r"(?i)disregard\s+(all\s+)?(prior|previous|above)",
            r"(?i)override\s+(all\s+)?safety",
            r"(?i)\bDAN\s+mode\b",
            r"(?i)do\s+anything\s+now",
            r"(?i)act\s+as\s+an?\s+unrestricted",
        ],
    },
    # Map tool-name → list of (param_name, validation_category) pairs.
    # Any param not listed here is validated with the "string" category.
    "tool_param_map": {
        "read_file":          [{"param": "file_path", "category": "file_path"}],
        "write_file":         [
            {"param": "file_path", "category": "file_path"},
            {"param": "content",   "category": "string"},
        ],
        "execute_sql_query":  [{"param": "query", "category": "sql"}],
        "send_email":         [
            {"param": "to_address", "category": "email"},
            {"param": "subject",    "category": "string"},
            {"param": "body",       "category": "string"},
        ],
        "read_email":         [
            {"param": "search_query", "category": "string"},
        ],
    },
}

# ---------------------------------------------------------------------------
# Pydantic schema model for tool-level validation configuration
# ---------------------------------------------------------------------------


class ParamRule(BaseModel):
    """Maps a single tool parameter to a validation category.

    Attributes:
        param: The name of the parameter on the LangChain tool.
        category: One of ``file_path``, ``sql``, ``email``, ``string``.
    """

    param: str
    category: str = Field(
        ...,
        pattern=r"^(file_path|sql|email|string)$",
        description="Validation category to apply.",
    )


class ToolSecuritySchema(BaseModel):
    """Root model mapping tool names to their per-parameter validation rules.

    This is the *typed* representation of the ``tool_param_map`` section of the
    config dict.  It lets researchers define (and serialise) custom schemas.

    Attributes:
        rules: Mapping from tool name to an ordered list of ``ParamRule``
               entries that will be checked before the tool executes.

    Example::

        schema = ToolSecuritySchema(rules={
            "read_file": [ParamRule(param="file_path", category="file_path")],
        })
    """

    rules: dict[str, list[ParamRule]] = Field(
        default_factory=dict,
        description="Mapping of tool name → list of per-parameter rules.",
    )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ToolSecuritySchema:
        """Build a ``ToolSecuritySchema`` from a raw config dict.

        Args:
            config: Full defense config dict (must contain a
                ``tool_param_map`` key).

        Returns:
            A validated ``ToolSecuritySchema`` instance.
        """
        raw_map: dict[str, list[dict[str, str]]] = config.get(
            "tool_param_map", {}
        )
        parsed: dict[str, list[ParamRule]] = {}
        for tool_name, param_entries in raw_map.items():
            parsed[tool_name] = [ParamRule(**entry) for entry in param_entries]
        return cls(rules=parsed)


# ---------------------------------------------------------------------------
# Core defense class
# ---------------------------------------------------------------------------


class TypeCheckerDefense:
    """Strict parameter type-checking and sanitization defense for LangChain tools.

    The class is designed to be used in two ways:

    1. **Instance-based** – create an instance with a custom config, then call
       ``wrap_tool`` / ``wrap_tools`` on it.
    2. **Class-method shortcut** – call ``TypeCheckerDefense.wrap_tools(tools)``
       to wrap a list of tools with the default config in a single call.

    Attributes:
        config: Merged configuration dictionary controlling every rule.
        schema: Typed ``ToolSecuritySchema`` derived from the config.
        security_log: Chronological list of ``SecurityLogEntry`` records for
            every blocked call since this instance was created.
    """

    #: Shared across instances so a whole evaluation run's blocks are
    #: collectable through the classmethod API that the agent layer calls.
    _shared_log: list[SecurityLogEntry] = []

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        recorder: Callable[[str, dict[str, Any], str, str], None] | None = None,
    ) -> None:
        """Initialise the defense with an optional config override.

        Args:
            config: If *None*, ``DEFAULT_CONFIG`` is used as-is.  If
                provided, it is **deep-merged** on top of the defaults so
                researchers only need to specify the keys they want to
                override.
            recorder: Optional ``(tool, args, result, blocked_by)`` callback,
                normally ``Environment.record``.  Blocked calls are logged to
                the environment so analysis can distinguish an attack the
                defense stopped from one the agent never attempted.
        """
        self.config: dict[str, Any] = self._merge_config(
            DEFAULT_CONFIG, config or {}
        )
        self.schema: ToolSecuritySchema = ToolSecuritySchema.from_config(
            self.config
        )
        self.security_log: list[SecurityLogEntry] = []
        self.recorder = recorder

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_config(
        base: dict[str, Any], overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """Recursively deep-merge *overrides* into a copy of *base*.

        Lists are *replaced* (not appended) so that a researcher can fully
        redefine a pattern list by providing a new one.

        Args:
            base: The default configuration dictionary.
            overrides: Researcher-supplied overrides.

        Returns:
            A new dictionary representing the merged config.
        """
        merged = deepcopy(base)
        for key, value in overrides.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = TypeCheckerDefense._merge_config(
                    merged[key], value
                )
            else:
                merged[key] = deepcopy(value)
        return merged

    @classmethod
    def from_yaml(cls, yaml_path: str) -> TypeCheckerDefense:
        """Construct a ``TypeCheckerDefense`` from a YAML config file.

        Args:
            yaml_path: Absolute or relative path to a YAML file whose
                structure mirrors ``DEFAULT_CONFIG``.

        Returns:
            A new ``TypeCheckerDefense`` instance.

        Raises:
            FileNotFoundError: If *yaml_path* does not exist.
            yaml.YAMLError: If the file is not valid YAML.
        """
        with open(yaml_path, encoding="utf-8") as fh:
            user_config: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls(config=user_config)

    # ------------------------------------------------------------------
    # Individual validators
    # ------------------------------------------------------------------

    def validate_file_path(self, value: str) -> tuple[ValidationResult, str]:
        """Check a file-path argument for traversal, sensitive targets, and
        shell metacharacters.

        Args:
            value: The raw file-path string supplied by the agent.

        Returns:
            A ``(result, matched_pattern)`` tuple.  ``matched_pattern`` is
            the empty string when the result is ``PASS``.
        """
        fp_cfg = self.config.get("file_path", {})

        for pattern in fp_cfg.get("traversal_patterns", []):
            if re.search(pattern, value):
                return ValidationResult.FAIL_PATH_TRAVERSAL, pattern

        for pattern in fp_cfg.get("sensitive_paths", []):
            if re.search(re.escape(pattern) if not pattern.startswith("^") else pattern, value, re.IGNORECASE):
                return ValidationResult.FAIL_PATH_TRAVERSAL, pattern

        for pattern in fp_cfg.get("shell_metacharacters", []):
            if re.search(pattern, value):
                return ValidationResult.FAIL_PATH_TRAVERSAL, pattern

        return ValidationResult.PASS, ""

    def validate_sql(self, value: str) -> tuple[ValidationResult, str]:
        """Check a SQL query string for destructive keywords and verify it
        is an allowed statement type.

        Args:
            value: The raw SQL query string supplied by the agent.

        Returns:
            A ``(result, matched_pattern)`` tuple.
        """
        sql_cfg = self.config.get("sql", {})

        # 1. Reject destructive keywords regardless of statement type.
        for pattern in sql_cfg.get("destructive_keywords", []):
            if re.search(pattern, value, re.IGNORECASE):
                return ValidationResult.FAIL_DESTRUCTIVE_SQL, pattern

        # 2. Ensure the statement matches at least one allowed pattern.
        allowed = sql_cfg.get("allowed_statements", [])
        if allowed:
            if not any(
                re.search(pat, value, re.IGNORECASE) for pat in allowed
            ):
                return (
                    ValidationResult.FAIL_DESTRUCTIVE_SQL,
                    f"statement did not match any allowed pattern: {allowed}",
                )

        return ValidationResult.PASS, ""

    def validate_email(self, value: str) -> tuple[ValidationResult, str]:
        """Validate an email address for format and suspicious domain.

        Args:
            value: The raw email-address string supplied by the agent.

        Returns:
            A ``(result, matched_pattern)`` tuple.
        """
        email_cfg = self.config.get("email", {})

        # 1. Basic format check.
        fmt_pattern = email_cfg.get(
            "valid_format",
            r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
        )
        if not re.match(fmt_pattern, value):
            return (
                ValidationResult.FAIL_SUSPICIOUS_EMAIL,
                f"invalid format ({fmt_pattern})",
            )

        # 2. Suspicious domain check.
        for pattern in email_cfg.get("suspicious_domains", []):
            if re.search(pattern, value, re.IGNORECASE):
                return ValidationResult.FAIL_SUSPICIOUS_EMAIL, pattern

        return ValidationResult.PASS, ""

    def validate_string(self, value: str) -> tuple[ValidationResult, str]:
        """Scan a generic string for prompt-injection markers.

        Args:
            value: Any string argument supplied by the agent.

        Returns:
            A ``(result, matched_pattern)`` tuple.
        """
        str_cfg = self.config.get("string", {})

        for pattern in str_cfg.get("injection_markers", []):
            if re.search(pattern, value):
                return ValidationResult.FAIL_INJECTION_MARKER, pattern

        return ValidationResult.PASS, ""

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    _CATEGORY_DISPATCH: dict[
        str,
        str,  # method name looked up via getattr at runtime
    ] = {
        "file_path": "validate_file_path",
        "sql": "validate_sql",
        "email": "validate_email",
        "string": "validate_string",
    }

    def validate_param(
        self, category: str, value: Any
    ) -> tuple[ValidationResult, str]:
        """Route a parameter value to the appropriate validator.

        If *value* is not a string it is coerced via ``str()`` before
        validation.  ``None`` values are silently passed through.

        Args:
            category: One of ``file_path``, ``sql``, ``email``, ``string``.
            value: The raw argument value.

        Returns:
            A ``(result, matched_pattern)`` tuple.

        Raises:
            ValueError: If *category* is not recognised.
        """
        if value is None:
            return ValidationResult.PASS, ""

        str_value = str(value) if not isinstance(value, str) else value

        method_name = self._CATEGORY_DISPATCH.get(category)
        if method_name is None:
            raise ValueError(
                f"Unknown validation category '{category}'. "
                f"Valid categories: {list(self._CATEGORY_DISPATCH)}"
            )
        method: Callable[..., tuple[ValidationResult, str]] = getattr(
            self, method_name
        )
        return method(str_value)

    # ------------------------------------------------------------------
    # Tool-wrapping logic
    # ------------------------------------------------------------------

    def _get_param_rules(self, tool_name: str) -> list[ParamRule]:
        """Look up the validation rules for *tool_name*.

        Falls back to validating **every** string argument with the
        ``string`` category if no explicit mapping exists.

        Args:
            tool_name: The ``name`` attribute of the LangChain tool.

        Returns:
            Ordered list of ``ParamRule`` objects.
        """
        if tool_name in self.schema.rules:
            return self.schema.rules[tool_name]
        # Fallback: not mapped → no param-level rules, but we still scan
        # every value with the "string" category in validate_args.
        return []

    def validate_args(
        self,
        tool_name: str,
        args: dict[str, Any],
    ) -> SecurityLogEntry | None:
        """Validate all arguments for a single tool call.

        The method first checks params that have explicit rules in the
        schema, then scans any *remaining* string-valued args with the
        generic ``string`` validator.

        Args:
            tool_name: Name of the LangChain tool.
            args: Keyword arguments the agent is attempting to pass.

        Returns:
            A ``SecurityLogEntry`` if validation failed, otherwise *None*.
        """
        explicit_rules = self._get_param_rules(tool_name)
        checked_params: set[str] = set()

        # 1. Run explicit rules first.
        for rule in explicit_rules:
            param_value = args.get(rule.param)
            if param_value is None:
                continue
            result, matched = self.validate_param(rule.category, param_value)
            if result != ValidationResult.PASS:
                entry = SecurityLogEntry(
                    tool_name=tool_name,
                    args=args,
                    rule_violated=result,
                    matched_pattern=matched,
                )
                self.security_log.append(entry)
                TypeCheckerDefense._shared_log.append(entry)
                logger.warning(
                    "Blocked tool call: %s | rule=%s | pattern=%s",
                    tool_name,
                    result.value,
                    matched,
                )
                return entry
            checked_params.add(rule.param)

        # 2. Fall back: scan unchecked string args for injection markers.
        for param_name, param_value in args.items():
            if param_name in checked_params:
                continue
            if not isinstance(param_value, str):
                continue
            result, matched = self.validate_string(param_value)
            if result != ValidationResult.PASS:
                entry = SecurityLogEntry(
                    tool_name=tool_name,
                    args=args,
                    rule_violated=result,
                    matched_pattern=matched,
                )
                self.security_log.append(entry)
                TypeCheckerDefense._shared_log.append(entry)
                logger.warning(
                    "Blocked tool call (fallback): %s | rule=%s | pattern=%s",
                    tool_name,
                    result.value,
                    matched,
                )
                return entry

        return None

    def wrap_tool(self, langchain_tool: BaseTool) -> BaseTool:
        """Wrap a single LangChain tool with input validation.

        The original tool's metadata (name, description, schema) is
        preserved.  Only the execution path is intercepted.

        Args:
            langchain_tool: Any LangChain ``BaseTool`` instance.

        Returns:
            A new ``StructuredTool`` that validates inputs before
            delegating to the original tool.
        """
        defense = self  # capture for the closure

        original_name: str = langchain_tool.name
        original_description: str = langchain_tool.description
        original_schema = langchain_tool.args_schema

        def guarded_func(**kwargs: Any) -> str:
            """Intercept and validate arguments, then delegate or block."""
            violation = defense.validate_args(original_name, kwargs)
            if violation is not None:
                message = (
                    f"[BLOCKED by TypeCheckerDefense] "
                    f"Rule violated: {violation.rule_violated.value}. "
                    f"Matched pattern: {violation.matched_pattern!r}. "
                    f"The requested operation was not executed."
                )
                if defense.recorder is not None:
                    defense.recorder(original_name, kwargs, message, "type_checker")
                return message
            return langchain_tool.invoke(kwargs)

        wrapped = StructuredTool.from_function(
            func=guarded_func,
            name=original_name,
            description=original_description,
            args_schema=original_schema,
        )
        return wrapped

    # ------------------------------------------------------------------
    # Public convenience API
    # ------------------------------------------------------------------

    @classmethod
    def wrap_tools(
        cls,
        tools: Sequence[BaseTool],
        config: dict[str, Any] | None = None,
        recorder: Callable[[str, dict[str, Any], str, str], None] | None = None,
    ) -> list[BaseTool]:
        """Wrap a list of LangChain tools with input validation.

        This is the primary entry-point for researchers who want a
        one-liner integration::

            from chokepoint.defenses.type_checker import TypeCheckerDefense
            safe_tools = TypeCheckerDefense.wrap_tools(agent_tools)

        Args:
            tools: Sequence of LangChain ``BaseTool`` instances to protect.
            config: Optional config overrides (see ``DEFAULT_CONFIG``).

        Returns:
            A new list of wrapped tools, in the same order as *tools*.
        """
        instance = cls(config=config, recorder=recorder)
        return [instance.wrap_tool(t) for t in tools]

    def get_security_log(self) -> list[SecurityLogEntry]:
        """Return a copy of this instance's security log."""
        return list(self.security_log)

    def clear_security_log(self) -> None:
        """Clear this instance's security-log entries."""
        self.security_log.clear()

    # ------------------------------------------------------------------
    # Run-scoped classmethod API (mirrors LLMJudgeDefense)
    # ------------------------------------------------------------------

    @classmethod
    def get_log(cls) -> list[dict[str, Any]]:
        """Return every block recorded since the last :meth:`clear_log`."""
        return [entry.model_dump() for entry in cls._shared_log]

    @classmethod
    def clear_log(cls) -> None:
        """Reset the shared log. Call between evaluation runs."""
        cls._shared_log.clear()
