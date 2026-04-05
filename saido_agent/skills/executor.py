"""Skill execution: inline (current conversation) or forked (sub-agent)."""
from __future__ import annotations

from typing import Generator

from .loader import SkillDef, substitute_arguments


def execute_skill(skill, args, state, config, system_prompt) -> Generator:
    rendered = substitute_arguments(skill.prompt, args, skill.arguments)
    message = f"[Skill: {skill.name}]\n\n{rendered}"

    if skill.context == "fork":
        yield from _execute_forked(skill, message, config, system_prompt)
    else:
        yield from _execute_inline(message, state, config, system_prompt)


def _execute_inline(message, state, config, system_prompt) -> Generator:
    from saido_agent.core import agent as _agent
    yield from _agent.run(message, state, config, system_prompt)


def _execute_forked(skill, message, config, system_prompt) -> Generator:
    from saido_agent.core import agent as _agent

    depth = config.get("_depth", 0) + 1
    sub_config = {**config, "_depth": depth, "_system_prompt": system_prompt}
    if skill.model:
        sub_config["model"] = skill.model
    if skill.tools:
        sub_config["_allowed_tools"] = skill.tools

    sub_state = _agent.AgentState()
    yield from _agent.run(message, sub_state, sub_config, system_prompt)
