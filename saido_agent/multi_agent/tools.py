"""Multi-agent tool registrations for Saido Agent."""
from __future__ import annotations

from saido_agent.core.tool_registry import ToolDef, register_tool
from .subagent import SubAgentManager, get_agent_definition, load_agent_definitions
from .resources import AgentResourceLimits


_agent_manager: SubAgentManager | None = None


def get_agent_manager() -> SubAgentManager:
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = SubAgentManager()
    return _agent_manager


def _agent_tool(params: dict, config: dict) -> str:
    mgr = get_agent_manager()
    prompt = params["prompt"]
    wait = params.get("wait", True)
    isolation = params.get("isolation", "")
    name = params.get("name", "")
    model_override = params.get("model", "")
    subagent_type = params.get("subagent_type", "")
    system_prompt = config.get("_system_prompt", "You are a helpful assistant.")
    depth = config.get("_depth", 0)
    eff_config = {k: v for k, v in config.items() if not k.startswith("_")}
    if model_override:
        eff_config["model"] = model_override
    agent_def = None
    if subagent_type:
        agent_def = get_agent_definition(subagent_type)
        if agent_def is None:
            return f"Error: unknown subagent_type '{subagent_type}'. Use ListAgentTypes to see available types."
    task = mgr.spawn(prompt, eff_config, system_prompt, depth=depth, agent_def=agent_def, isolation=isolation, name=name)
    if task.status == "failed":
        return f"Error spawning agent: {task.result}"
    if wait:
        mgr.wait(task.id, timeout=300)
        result = task.result or f"(no output -- status: {task.status})"
        header = f"[Agent: {task.name}"
        if subagent_type:
            header += f" ({subagent_type})"
        if task.worktree_branch:
            header += f", branch: {task.worktree_branch}"
        header += "]"
        # Append resource usage summary if available
        resource_footer = ""
        if task.resource_summary:
            rs = task.resource_summary
            resource_footer = (
                f"\n\n[Resources: {rs.get('tokens_used', 0)} tokens, "
                f"{rs.get('turns_used', 0)} turns, "
                f"{rs.get('tool_calls_used', 0)} tool calls, "
                f"{rs.get('elapsed_seconds', 0)}s elapsed"
            )
            if rs.get("exceeded_limit"):
                resource_footer += f" | EXCEEDED: {rs['exceeded_limit']}"
            resource_footer += "]"
        return f"{header}\n\n{result}{resource_footer}"
    else:
        info_parts = [f"Task ID: {task.id}", f"Name: {task.name}", f"Status: {task.status}"]
        if subagent_type:
            info_parts.append(f"Type: {subagent_type}")
        if task.worktree_branch:
            info_parts.append(f"Worktree branch: {task.worktree_branch}")
        info_parts.append("Use CheckAgentResult or SendMessage to interact with this agent.")
        return "\n".join(info_parts)


def _send_message(params: dict, config: dict) -> str:
    mgr = get_agent_manager()
    target = params["to"]
    message = params["message"]
    ok = mgr.send_message(target, message)
    if ok:
        return f"Message queued for agent '{target}'."
    task_id = mgr._by_name.get(target, target)
    task = mgr.tasks.get(task_id)
    if task is None:
        return f"Error: no agent found with id or name '{target}'"
    return f"Error: agent '{target}' is not running (status: {task.status})."


def _check_agent_result(params: dict, config: dict) -> str:
    mgr = get_agent_manager()
    task_id = params["task_id"]
    task = mgr.tasks.get(task_id)
    if task is None:
        return f"Error: no task with id '{task_id}'"
    lines = [f"Status: {task.status}", f"Name: {task.name}"]
    if task.worktree_branch:
        lines.append(f"Worktree branch: {task.worktree_branch}")
    if task.result:
        lines.append(f"\nResult:\n{task.result}")
    return "\n".join(lines)


def _list_agent_tasks(params: dict, config: dict) -> str:
    mgr = get_agent_manager()
    tasks = mgr.list_tasks()
    if not tasks:
        return "No sub-agent tasks."
    lines = ["ID           | Name     | Status    | Worktree branch | Prompt"]
    lines.append("-------------|----------|-----------|-----------------|------")
    for t in tasks:
        prompt_short = t.prompt[:50] + ("..." if len(t.prompt) > 50 else "")
        wt = t.worktree_branch[:15] if t.worktree_branch else "-"
        lines.append(f"{t.id} | {t.name[:8]:8s} | {t.status:9s} | {wt:15s} | {prompt_short}")
    return "\n".join(lines)


def _list_agent_types(params: dict, config: dict) -> str:
    defs = load_agent_definitions()
    if not defs:
        return "No agent types available."
    lines = ["Available agent types:", ""]
    for aname, d in sorted(defs.items()):
        model_info = f"  model: {d.model}" if d.model else ""
        tools_info = f"  tools: {', '.join(d.tools)}" if d.tools else ""
        lines.append(f"  {aname:20s}  [{d.source:8s}]  {d.description}")
        if model_info:
            lines.append(f"                           {model_info}")
        if tools_info:
            lines.append(f"                           {tools_info}")
    lines.append("")
    lines.append("Create custom agents: place .md files in ~/.saido_agent/agents/ or .saido_agent/agents/")
    return "\n".join(lines)


register_tool(ToolDef(name="Agent", schema={"name": "Agent", "description": "Spawn a sub-agent to handle a task autonomously.", "input_schema": {"type": "object", "properties": {"prompt": {"type": "string", "description": "Task description"}, "subagent_type": {"type": "string", "description": "Specialized agent type"}, "name": {"type": "string", "description": "Human-readable name"}, "model": {"type": "string", "description": "Model override"}, "wait": {"type": "boolean", "description": "Block until complete (default: true)"}, "isolation": {"type": "string", "enum": ["worktree"], "description": "Isolation mode"}}, "required": ["prompt"]}}, func=_agent_tool, read_only=False, concurrent_safe=False))
register_tool(ToolDef(name="SendMessage", schema={"name": "SendMessage", "description": "Send a follow-up message to a running background agent.", "input_schema": {"type": "object", "properties": {"to": {"type": "string", "description": "Agent name or task ID"}, "message": {"type": "string", "description": "Message to send"}}, "required": ["to", "message"]}}, func=_send_message, read_only=False, concurrent_safe=True))
register_tool(ToolDef(name="CheckAgentResult", schema={"name": "CheckAgentResult", "description": "Check status/result of a spawned sub-agent task.", "input_schema": {"type": "object", "properties": {"task_id": {"type": "string", "description": "Task ID"}}, "required": ["task_id"]}}, func=_check_agent_result, read_only=True, concurrent_safe=True))
register_tool(ToolDef(name="ListAgentTasks", schema={"name": "ListAgentTasks", "description": "List all sub-agent tasks.", "input_schema": {"type": "object", "properties": {}}}, func=_list_agent_tasks, read_only=True, concurrent_safe=True))
register_tool(ToolDef(name="ListAgentTypes", schema={"name": "ListAgentTypes", "description": "List all available agent types.", "input_schema": {"type": "object", "properties": {}}}, func=_list_agent_types, read_only=True, concurrent_safe=True))
