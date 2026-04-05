"""Multi-agent package for Saido Agent."""
from .subagent import (
    AgentDefinition,
    SubAgentTask,
    SubAgentManager,
    load_agent_definitions,
    get_agent_definition,
)
from .resources import (
    AgentResourceLimits,
    AgentResourceUsage,
    ResourceTracker,
)
from .messaging import (
    AgentInbox,
    AgentMessage,
)

__all__ = [
    "AgentDefinition",
    "SubAgentTask",
    "SubAgentManager",
    "load_agent_definitions",
    "get_agent_definition",
    "AgentResourceLimits",
    "AgentResourceUsage",
    "ResourceTracker",
    "AgentInbox",
    "AgentMessage",
]
