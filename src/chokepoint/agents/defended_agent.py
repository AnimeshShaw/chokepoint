"""Defended agent: interposition at the tool-call chokepoint.

Wraps a supplied tool list in one or more defenses, then delegates to
:class:`~chokepoint.agents.base_agent.BaseAgent`. Every defense is a mediator
between the model's proposed call and the tool implementation, which is the one
point in the loop where a policy can be enforced regardless of what the model
was persuaded to believe.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

from langchain_core.tools import BaseTool

from chokepoint.agents.base_agent import BaseAgent


class DefenseType(str, Enum):
    """Available interposition strategies."""

    NONE = "none"
    TYPE_CHECKER = "type_checker"
    CAPABILITY_ROUTER = "capability_router"
    LLM_JUDGE = "llm_judge"
    JUDGE_AND_TYPE = "llm_judge+type_checker"
    ALL = "all"


_COMPOSITES: dict[DefenseType, list[DefenseType]] = {
    DefenseType.NONE: [],
    DefenseType.JUDGE_AND_TYPE: [DefenseType.TYPE_CHECKER, DefenseType.LLM_JUDGE],
    DefenseType.ALL: [
        DefenseType.CAPABILITY_ROUTER,
        DefenseType.TYPE_CHECKER,
        DefenseType.LLM_JUDGE,
    ],
}


class DefendedAgent:
    """A :class:`BaseAgent` whose tools are mediated by one or more defenses."""

    def __init__(
        self,
        model_id: str,
        tools: list[BaseTool],
        defense_type: DefenseType = DefenseType.NONE,
        judge_model: str | None = None,
        intent_strategy: str = "keyword",
        capability_map: dict | None = None,
        max_iterations: int = 10,
        verbose: bool = False,
        recorder: Callable[[str, dict[str, Any], str, str], None] | None = None,
    ) -> None:
        """
        Args:
            recorder: Optional ``Environment.record`` callback so that calls a
                defense blocks still appear in the execution log, marked with
                the blocking layer. Without it, an attack the defense stopped
                is indistinguishable from one the agent never attempted.
        """
        self.model_id = model_id
        self.tools = tools
        self.defense_type = defense_type
        self.judge_model = judge_model or model_id
        self.recorder = recorder
        self.intent_strategy = intent_strategy
        self.capability_map = capability_map
        self.max_iterations = max_iterations
        self.verbose = verbose
        self._router = None

    @property
    def active_defenses(self) -> list[DefenseType]:
        """Resolve a composite defense into its individual layers."""
        if self.defense_type in _COMPOSITES:
            return _COMPOSITES[self.defense_type]
        return [self.defense_type]

    def _apply_defenses(self, tools: list[BaseTool], user_prompt: str) -> list[BaseTool]:
        from chokepoint.defenses.capability_router import (
            CapabilityRouterConfig,
            CapabilityRouterDefense,
        )
        from chokepoint.defenses.llm_judge import LLMJudgeDefense
        from chokepoint.defenses.type_checker import TypeCheckerDefense

        wrapped = tools
        for defense in self.active_defenses:
            if defense is DefenseType.TYPE_CHECKER:
                TypeCheckerDefense.clear_log()
                wrapped = TypeCheckerDefense.wrap_tools(wrapped, recorder=self.recorder)
            elif defense is DefenseType.LLM_JUDGE:
                LLMJudgeDefense.clear_log()
                wrapped = LLMJudgeDefense.wrap_tools(
                    wrapped,
                    user_prompt=user_prompt,
                    judge_model=self.judge_model,
                    recorder=self.recorder,
                )
            elif defense is DefenseType.CAPABILITY_ROUTER:
                self._router = CapabilityRouterDefense(
                    config=CapabilityRouterConfig(
                        classification_strategy=self.intent_strategy,
                        capability_map=self.capability_map,
                        classifier_model_name=self.model_id,
                    )
                )
                wrapped = self._router.get_allowed_tools(user_prompt, wrapped)
        return wrapped

    def run(self, user_prompt: str) -> dict[str, Any]:
        """Apply defenses for this prompt, then execute one episode."""
        defended = self._apply_defenses(self.tools, user_prompt)

        agent = BaseAgent(
            model_id=self.model_id,
            tools=defended,
            max_iterations=self.max_iterations,
            verbose=self.verbose,
        )
        result = agent.run(user_prompt)
        result["defense_type"] = self.defense_type.value
        result["defense_logs"] = self._collect_logs()
        return result

    def _collect_logs(self) -> dict[str, Any]:
        """Gather each active defense's decision record.

        Interface errors are deliberately not caught here. The pre-audit
        version wrapped this in a bare ``except``, which silently hid a call to
        a non-existent ``TypeCheckerDefense.get_log`` and left every
        type-checker log empty across the whole evaluation.
        """
        from chokepoint.defenses.llm_judge import LLMJudgeDefense
        from chokepoint.defenses.type_checker import TypeCheckerDefense

        logs: dict[str, Any] = {}
        active = self.active_defenses

        if DefenseType.TYPE_CHECKER in active:
            logs["type_checker"] = TypeCheckerDefense.get_log()
        if DefenseType.LLM_JUDGE in active:
            logs["llm_judge"] = LLMJudgeDefense.get_log()
        if DefenseType.CAPABILITY_ROUTER in active and self._router is not None:
            logs["capability_router"] = self._router.get_log_summary()

        return logs

    def info(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "agent_type": "defended",
            "defense": self.defense_type.value,
            "layers": [d.value for d in self.active_defenses],
            "judge_model": self.judge_model,
        }
