"""Knowledge system for Saido Agent — bridge to SmartRAG retrieval engine."""

from saido_agent.knowledge.bridge import (
    SMARTRAG_AVAILABLE,
    BridgeConfig,
    KnowledgeBridge,
)

__all__ = [
    "BridgeConfig",
    "KnowledgeBridge",
    "SMARTRAG_AVAILABLE",
]
