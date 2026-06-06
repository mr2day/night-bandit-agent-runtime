---
name: deep_research
description: Thorough, multi-source, fact-checked research on a complex topic. Use when the user wants depth and verified sources rather than a quick answer.
---

# Deep Research playbook

Follow these steps to produce a thorough, verified answer:

1. **Decompose** the question into 3-5 specific sub-questions.
2. **Search broadly**: run `search_web` for each sub-question with short
   (3-6 word) queries. Prefer authoritative and primary sources.
3. **Read the best sources**: for the 3-5 most promising results, call
   `fetch_page` to read the full text — never answer from snippets alone
   when depth matters.
4. **Cross-check**: where claims disagree across sources, note the
   disagreement explicitly rather than picking one silently.
5. **Synthesize**: write the answer grounded in what you read. Attribute
   key claims to their sources (title + URL).
6. **State uncertainty**: if the sources are thin or conflicting, say so.

Do not pad the answer. Length should match the depth the evidence supports.
