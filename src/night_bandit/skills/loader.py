"""Load SKILL.md playbooks and expose them via a single ``use_skill`` tool.

A skill is a directory ``skills/<name>/SKILL.md`` with YAML frontmatter
(``name``, ``description``) and a markdown body of instructions. When the
agent calls ``use_skill(name)``, it gets the body back as the tool result
— a playbook it then follows using its other tools. This mirrors the
Anthropic "skills are model-followed instructions" model; an optional
``handler.py`` hook is left as a future extension point.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic_ai import Tool


@dataclass
class Skill:
    name: str
    description: str
    body: str


def _parse_skill_md(text: str, *, fallback_name: str) -> Skill:
    name = fallback_name
    description = ""
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            front = text[3:end]
            body = text[end + 4 :].lstrip("\n")
            try:
                meta = yaml.safe_load(front) or {}
                if isinstance(meta, dict):
                    name = str(meta.get("name", fallback_name))
                    description = str(meta.get("description", ""))
            except yaml.YAMLError:
                pass
    return Skill(name=name, description=description.strip(), body=body.strip())


def load_skills(skills_dir: str | Path) -> list[Skill]:
    """Scan ``skills_dir`` for ``*/SKILL.md`` and return parsed skills."""
    root = Path(skills_dir)
    if not root.is_dir():
        return []
    skills: list[Skill] = []
    for skill_md in sorted(root.glob("*/SKILL.md")):
        text = skill_md.read_text(encoding="utf-8")
        skills.append(_parse_skill_md(text, fallback_name=skill_md.parent.name))
    return skills


def build_use_skill_tool(skills: list[Skill]) -> Tool | None:
    """Build the ``use_skill`` tool whose description advertises the catalog.

    Returns None when there are no skills (so the agent isn't given an
    empty capability).
    """
    if not skills:
        return None

    by_name = {s.name: s for s in skills}
    catalog = "\n".join(f"  - {s.name}: {s.description}" for s in skills)

    async def use_skill(skill_name: str) -> dict[str, Any]:
        skill = by_name.get(skill_name)
        if skill is None:
            return {
                "error": f"unknown skill '{skill_name}'",
                "available": list(by_name.keys()),
            }
        return {"skill": skill.name, "instructions": skill.body}

    description = (
        "Invoke a named skill to get a step-by-step playbook for a complex "
        "task, then follow those instructions using your other tools. "
        "Available skills:\n" + catalog
    )
    return Tool(use_skill, takes_ctx=False, description=description)
