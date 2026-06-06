"""Built-in tools: get_current_time, search_web, fetch_page, browse.

Each is a typed async function; Pydantic AI derives the JSON schema from
the signature + docstring. External-dependency tools (Exa, Playwright)
import lazily so the package loads even when those aren't configured —
they raise a clear error only if actually called without setup.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic_ai import Tool

from ..config import get_settings


async def get_current_time(timezone: str = "UTC") -> dict[str, Any]:
    """Get the current date and time in the given IANA timezone.

    Args:
        timezone: IANA timezone name, e.g. "Europe/Bucharest". Defaults to UTC.
    """
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        tz = ZoneInfo("UTC")
        timezone = "UTC"
    now = datetime.now(tz)
    return {
        "timezone": timezone,
        "iso": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
    }


def _freshness_to_start_date(freshness: str) -> str | None:
    if freshness == "any":
        return None
    from datetime import timedelta, timezone as _tz

    days = {"day": 1, "week": 7, "month": 31, "year": 366}.get(freshness, 31)
    start = datetime.now(_tz.utc) - timedelta(days=days)
    return start.isoformat()


async def search_web(
    query: str,
    freshness: Literal["day", "week", "month", "year", "any"] = "month",
    num_results: int = 8,
) -> dict[str, Any]:
    """Search the live web. Returns titles, URLs, dates, and ~600-char snippets.

    Use short, natural queries (3-6 words); the engine is semantic and
    ignores boolean/site: operators and quotes. Use for recent events,
    current-state questions, and things you can't answer confidently.

    Args:
        query: The natural-language search query.
        freshness: How recent results must be.
        num_results: How many results (1-20).
    """
    settings = get_settings()
    from exa_py import Exa

    exa = Exa(settings.exa_api_key)
    start = _freshness_to_start_date(freshness)
    num = max(1, min(20, num_results))

    def _run() -> Any:
        kwargs: dict[str, Any] = {
            "num_results": num,
            "type": "auto",
            "text": {"max_characters": 600},
        }
        if start:
            kwargs["start_published_date"] = start
        return exa.search_and_contents(query, **kwargs)

    resp = await asyncio.to_thread(_run)
    results = []
    for r in getattr(resp, "results", []) or []:
        text = getattr(r, "text", None)
        results.append(
            {
                "title": getattr(r, "title", "") or "",
                "url": getattr(r, "url", ""),
                "published_date": getattr(r, "published_date", None),
                "snippet": (text or "")[:600],
            }
        )
    return {"query": query, "results": results}


async def fetch_page(url: str) -> dict[str, Any]:
    """Fetch the full text of a single URL (live-crawled).

    Use after search_web when a snippet was cut off before the answer, or
    for a known canonical URL.

    Args:
        url: The exact URL to fetch.
    """
    settings = get_settings()
    from exa_py import Exa

    exa = Exa(settings.exa_api_key)

    def _run() -> Any:
        return exa.get_contents([url], text=True, livecrawl="always")

    resp = await asyncio.to_thread(_run)
    results = getattr(resp, "results", []) or []
    if not results:
        return {"url": url, "title": None, "content": "", "content_length": 0, "truncated": False}
    first = results[0]
    raw = getattr(first, "text", "") or ""
    cap = settings.fetch_page_max_chars
    truncated = len(raw) > cap
    return {
        "url": getattr(first, "url", url),
        "title": getattr(first, "title", None),
        "published_date": getattr(first, "published_date", None),
        "content": raw[:cap] if truncated else raw,
        "content_length": len(raw),
        "truncated": truncated,
    }


async def browse(url: str, full_page: bool = False, wait_ms: int = 500) -> dict[str, Any]:
    """Load a URL in a real headless browser and return rendered text + links.

    Use for JS-heavy pages where fetch_page returns thin content, or when
    you need the rendered structure. Returns extracted body text and the
    anchor links found on the page.

    Args:
        url: The URL to load (http/https).
        full_page: Render the whole document (true) or just the viewport.
        wait_ms: Extra wait after network idle, in ms.
    """
    settings = get_settings()
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            context = await browser.new_context()
            page = await context.new_page()
            resp = await page.goto(
                url, wait_until="networkidle", timeout=settings.browse_nav_timeout_ms
            )
            status = resp.status if resp else 0
            await page.wait_for_timeout(min(max(wait_ms, 0), 10000))
            title = await page.title()
            text = await page.evaluate("() => document.body.innerText")
            links = await page.evaluate(
                """(limit) => Array.from(document.querySelectorAll('a[href]'))
                    .slice(0, limit)
                    .map(a => ({ href: a.href, text: (a.innerText||'').trim().slice(0,100) }))""",
                settings.browse_max_links,
            )
            cap = settings.browse_max_text_chars
            truncated = len(text) > cap
            return {
                "url": page.url,
                "title": title or None,
                "status": status,
                "text": text[:cap] if truncated else text,
                "text_length": len(text),
                "text_truncated": truncated,
                "links": links,
            }
        finally:
            await browser.close()


def builtin_tools() -> list[Tool]:
    """The built-in tools as Pydantic AI Tool objects (no RunContext needed).

    Pass to ``Agent(tools=builtin_tools())`` at construction.
    """
    return [
        Tool(get_current_time, takes_ctx=False),
        Tool(search_web, takes_ctx=False),
        Tool(fetch_page, takes_ctx=False),
        Tool(browse, takes_ctx=False),
    ]
