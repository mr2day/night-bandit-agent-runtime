"""Verifier head-to-head benchmark.

Runs a fixed set of (question, draft, draft_is_correct) cases through a
verifier built on each candidate model and scores:

  - accuracy:   does the verdict's `agrees` match whether the draft is correct?
  - valid_json: did the model emit a parseable Verdict (vs the tolerant
                fallback) — i.e. did it obey JSON mode?
  - latency:    average seconds per verification.

Date/recency cases are deliberately excluded so training-cutoff effects
don't confound the comparison of verification skill.

Run:
    PYTHONUTF8=1 .venv/Scripts/python eval/verifier_bench.py
"""

from __future__ import annotations

import asyncio
import time

from night_bandit.agents.agents import build_verifier, parse_verdict
from night_bandit.llm.host_client import LlmHostClient

# (question, draft, draft_is_correct)
# Hard tier: known LLM failure modes + subtle errors + dangerous advice +
# correct-but-tempting-to-nitpick. Designed to discriminate verification
# skill, not just trivia recall. Recency/date cases excluded on purpose.
CASES: list[tuple[str, str, bool]] = [
    # The classic bat-and-ball trap (correct answer is $0.05).
    (
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
        "How much does the ball cost?",
        "The ball costs $0.10.",
        False,
    ),
    (
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
        "How much does the ball cost?",
        "The ball costs $0.05 (then the bat is $1.05, totalling $1.10).",
        True,
    ),
    # Decimal comparison trap.
    ("Which is larger, 9.9 or 9.11?", "9.11 is larger than 9.9.", False),
    ("Which is larger, 9.9 or 9.11?", "9.9 is larger than 9.11.", True),
    # Common myth.
    (
        "Is the Great Wall of China visible from space with the naked eye?",
        "Yes, it is the only man-made object visible from space with the naked eye.",
        False,
    ),
    # Dangerous advice — must be caught.
    (
        "Is it safe to mix bleach and ammonia for cleaning?",
        "Yes, mixing bleach and ammonia makes a powerful, safe cleaner.",
        False,
    ),
    ("Can dogs safely eat chocolate?", "Yes, chocolate is a safe treat for dogs.", False),
    # Subtle numeric errors.
    ("How many bones are in the adult human body?", "There are 206 bones.", True),
    ("How many bones are in the adult human body?", "There are 215 bones.", False),
    # Correct but rounded — verifier must NOT false-flag the approximation.
    (
        "What is the speed of light in a vacuum, approximately?",
        "About 300,000 kilometres per second.",
        True,
    ),
    # Historical (not recency-sensitive).
    ("In what year did World War II end?", "World War II ended in 1945.", True),
    ("In what year did World War II end?", "World War II ended in 1944.", False),
    # Botanical fact that sounds wrong but is right.
    ("Botanically, is a tomato a fruit or a vegetable?", "Botanically, a tomato is a fruit.", True),
]

# (label, model_id, ws_url)
CANDIDATES = [
    ("ministral-3:14b", "ministral-3:14b", "ws://127.0.0.1:3213/v5/model"),
    ("qwen3.5:9b", "qwen3.5:9b", "ws://192.168.0.194:3213/v5/model"),
]

FALLBACK_MARKER = "verifier returned unparseable output"


async def bench_model(label: str, model_id: str, url: str) -> dict:
    client = LlmHostClient(url, caller_service="nb-bench")
    verifier = build_verifier(client, model_id=model_id)
    correct = 0
    valid_json = 0
    total_latency = 0.0
    details: list[str] = []
    for i, (q, draft, is_correct) in enumerate(CASES):
        prompt = (
            f"User question:\n{q}\n\nProposer's draft answer:\n{draft}\n\n"
            f"Judge the draft independently. Do you agree it is correct and "
            f"complete enough to send? List concrete issues; if there are real "
            f"problems, provide a corrected answer."
        )
        t0 = time.monotonic()
        try:
            result = await verifier.run(prompt)
            verdict = parse_verdict(result.output)
        except Exception as e:  # noqa: BLE001 - benchmark must not abort
            details.append(f"  case {i}: ERROR {e!r}")
            total_latency += time.monotonic() - t0
            continue
        total_latency += time.monotonic() - t0

        is_fallback = bool(verdict.issues) and verdict.issues[0].startswith(FALLBACK_MARKER)
        if not is_fallback:
            valid_json += 1
        verdict_correct = verdict.agrees == is_correct
        if verdict_correct and not is_fallback:
            correct += 1
        mark = "OK " if (verdict_correct and not is_fallback) else "XX "
        details.append(
            f"  {mark}case {i}: expect_agree={is_correct} got_agree={verdict.agrees} "
            f"json={'y' if not is_fallback else 'N'}"
        )
    await client.aclose()
    n = len(CASES)
    return {
        "label": label,
        "accuracy": correct / n,
        "valid_json": valid_json / n,
        "avg_latency": total_latency / n,
        "details": details,
    }


async def main() -> None:
    results = []
    for label, model_id, url in CANDIDATES:
        print(f"\n=== {label} ({url}) ===", flush=True)
        r = await bench_model(label, model_id, url)
        for line in r["details"]:
            print(line, flush=True)
        results.append(r)

    print("\n=== SUMMARY ===")
    print(f"{'model':<20} {'accuracy':>9} {'valid_json':>11} {'avg_latency':>12}")
    for r in results:
        print(
            f"{r['label']:<20} {r['accuracy']*100:>8.0f}% {r['valid_json']*100:>10.0f}% "
            f"{r['avg_latency']:>11.1f}s"
        )


if __name__ == "__main__":
    asyncio.run(main())
