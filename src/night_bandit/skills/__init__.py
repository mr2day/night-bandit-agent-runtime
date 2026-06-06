"""Skills: named markdown playbooks the agent can invoke via the
``use_skill`` tool. A skill is a SKILL.md (YAML frontmatter + markdown
instructions) under a skills directory."""

from .loader import Skill, build_use_skill_tool, load_skills

__all__ = ["Skill", "load_skills", "build_use_skill_tool"]
