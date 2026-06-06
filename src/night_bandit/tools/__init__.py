"""Built-in tools the agents can call. Pass to an agent at construction
via ``Agent(tools=builtin_tools())``."""

from .builtins import (
    browse,
    builtin_tools,
    fetch_page,
    get_current_time,
    search_web,
)

__all__ = [
    "builtin_tools",
    "get_current_time",
    "search_web",
    "fetch_page",
    "browse",
]
