---
name: check_current_info
description: Verify time-sensitive or "right now" claims against live sources before answering. Use for questions about current dates, prices, schedules, news, or anything that changes over time.
---

# Check current info playbook

For anything time-sensitive, do not answer from training memory:

1. If the question depends on the current date/time, call
   `get_current_time` with the user's timezone first.
2. For current facts (prices, schedules, news, who currently holds a role,
   "latest" anything), call `search_web` with a short, recent-focused
   query, then `fetch_page` on the most authoritative result to confirm.
3. Ground the answer in the tool results, and cite the source.
4. If a tool result and your prior belief conflict, trust the tool — your
   training cutoff may be older than today.
