"""Knowledge system for Saido Agent — bridge to SmartRAG retrieval engine."""

from saido_agent.knowledge.bridge import (
    HAS_EMBEDDINGS,
    SMARTRAG_AVAILABLE,
    BridgeConfig,
    KnowledgeBridge,
)
from saido_agent.knowledge.grounding import KnowledgeGrounder

__all__ = [
    "BridgeConfig",
    "HAS_EMBEDDINGS",
    "KnowledgeBridge",
    "KnowledgeGrounder",
    "SMARTRAG_AVAILABLE",
]
