"""Skill loading: parse markdown files with YAML frontmatter into SkillDef objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SkillDef:
    name: str
    description: str
    triggers: list[str]
    tools: list[str]
    prompt: str
    file_path: str
    when_to_use: str = ""
    argument_hint: str = ""
    arguments: list[str] = field(default_factory=list)
    model: str = ""
    user_invocable: bool = True
    context: str = "inline"
    source: str = "user"


def _get_skill_paths() -> list[Path]:
    return [
        Path.cwd() / ".saido_agent" / "skills",
        Path.home() / ".saido_agent" / "skills",
    ]


def _parse_list_field(value: str) -> list[str]:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [item.strip().strip('"').strip("'") for item in value.split(",") if item.strip()]


def _parse_skill_file(path: Path, source: str = "user") -> Optional[SkillDef]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    if not text.startswith("---"):
        return None

    parts = text.split("---", 2)
    if len(parts) < 3:
        return None

    frontmatter_raw = parts[1].strip()
    prompt = parts[2].strip()

    fields: dict[str, str] = {}
    for line in frontmatter_raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, val = line.partition(":")
        fields[key.strip().lower()] = val.strip()

    name = fields.get("name", "")
    if not name:
        return None

    tools_raw = fields.get("allowed-tools", fields.get("tools", ""))
    tools = _parse_list_field(tools_raw) if tools_raw else []

    triggers_raw = fields.get("triggers", "")
    triggers = _parse_list_field(triggers_raw) if triggers_raw else [f"/{name}"]

    arguments_raw = fields.get("arguments", "")
    arguments = _parse_list_field(arguments_raw) if arguments_raw else []

    user_invocable_raw = fields.get("user-invocable", "true")
    user_invocable = user_invocable_raw.lower() not in ("false", "0", "no")

    context = fields.get("context", "inline").strip().lower()
    if context not in ("inline", "fork"):
        context = "inline"

    return SkillDef(
        name=name, description=fields.get("description", ""),
        triggers=triggers, tools=tools, prompt=prompt, file_path=str(path),
        when_to_use=fields.get("when_to_use", ""),
        argument_hint=fields.get("argument-hint", ""),
        arguments=arguments, model=fields.get("model", ""),
        user_invocable=user_invocable, context=context, source=source,
    )


_BUILTIN_SKILLS: list[SkillDef] = []


def register_builtin_skill(skill: SkillDef) -> None:
    _BUILTIN_SKILLS.append(skill)


def load_skills(include_builtins: bool = True) -> list[SkillDef]:
    seen: dict[str, SkillDef] = {}
    if include_builtins:
        for sk in _BUILTIN_SKILLS:
            seen[sk.name] = sk
    skill_paths = _get_skill_paths()
    for i, skill_dir in enumerate(reversed(skill_paths)):
        src = "user" if i == 0 else "project"
        if not skill_dir.is_dir():
            continue
        for md_file in sorted(skill_dir.glob("*.md")):
            skill = _parse_skill_file(md_file, source=src)
            if skill:
                seen[skill.name] = skill
    return list(seen.values())


def find_skill(query: str) -> Optional[SkillDef]:
    query = query.strip()
    if not query:
        return None
    first_word = query.split()[0]
    for skill in load_skills():
        for trigger in skill.triggers:
            if first_word == trigger:
                return skill
            if trigger.startswith(first_word + " "):
                return skill
    return None


def substitute_arguments(prompt: str, args: str, arg_names: list[str]) -> str:
    result = prompt.replace("$ARGUMENTS", args)
    arg_values = args.split()
    for i, arg_name in enumerate(arg_names):
        placeholder = f"${arg_name.upper()}"
        value = arg_values[i] if i < len(arg_values) else ""
        result = result.replace(placeholder, value)
    return result
