"""The Night Bandit agent runtime.

A capable, local-only, two-LLM ensemble agent. LangGraph owns the
orchestration spine (durable execution, supervisor routing, human-in-the-
loop); Pydantic AI defines the typed agent nodes; both models are served
over the swirlock-llm-host v5 WebSocket protocol — proposer on Box A,
verifier on Box B.
"""

__version__ = "0.1.0"
