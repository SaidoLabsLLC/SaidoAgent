"""Knowledge system for Saido Agent — bridge to SmartRAG retrieval engine."""

from saido_agent.knowledge.bridge import (
    HAS_EMBEDDINGS,
    SMARTRAG_AVAILABLE,
    BridgeConfig,
    KnowledgeBridge,
)

__all__ = [
    "BridgeConfig",
    "HAS_EMBEDDINGS",
    "KnowledgeBridge",
    "SMARTRAG_AVAILABLE",
]
