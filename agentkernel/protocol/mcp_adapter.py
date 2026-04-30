"""AgentKernel — MCP (Model Context Protocol) native adapter.

MCP is treated as a first-class citizen, not an external adapter. This module
provides bidirectional MCP client/server capabilities integrated with the
Actor message-passing semantics.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID, uuid4

import agentkernel.types as _kt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MCP Server integration (AgentKernel exposes tools via MCP)
# ---------------------------------------------------------------------------

@dataclass
class MCPTool:
    """Internal representation of an MCP-compatible tool."""
    name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    handler: Optional[Callable[[Dict[str, Any]], Any]] = None
    cost_hint: float = 0.0  # Estimated USD cost per call
    isolation_hint: _kt.IsolationLevel = _kt.IsolationLevel.CONTAINER


class MCPToolRegistry:
    """Registry of locally-hosted MCP tools.

    The registry acts as an MCP *server* from the protocol perspective: it
    advertises capabilities and dispatches tool invocations. In AgentKernel
    terms, each registered tool is backed by an Actor or a pure function.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, MCPTool] = {}
        self._lock: asyncio.Lock = asyncio.Lock()

    async def register(self, tool: MCPTool) -> None:
        async with self._lock:
            self._tools[tool.name] = tool
        logger.info("MCP tool registered: %s", tool.name)

    async def unregister(self, tool_name: str) -> None:
        async with self._lock:
            self._tools.pop(tool_name, None)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Return MCP-compatible tool descriptors."""
        async with self._lock:
            return [
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.input_schema,
                }
                for t in self._tools.values()
            ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool by name with JSON-RPC-like semantics."""
        async with self._lock:
            tool = self._tools.get(tool_name)
        if tool is None:
            return {
                "content": [{"type": "text", "text": f"Tool '{tool_name}' not found"}],
                "isError": True,
            }
        if tool.handler is None:
            return {
                "content": [{"type": "text", "text": f"Tool '{tool_name}' has no handler"}],
                "isError": True,
            }
        try:
            result = tool.handler(arguments)
            if asyncio.iscoroutine(result):
                result = await result
            return {
                "content": [{"type": "text", "text": str(result)}],
                "isError": False,
            }
        except Exception as exc:
            logger.exception("MCP tool %s failed", tool_name)
            return {
                "content": [{"type": "text", "text": f"Error: {exc}"}],
                "isError": True,
            }


# ---------------------------------------------------------------------------
# MCP Client integration (AgentKernel calls external MCP servers)
# ---------------------------------------------------------------------------

@dataclass
class MCPConnection:
    """Lightweight connection metadata for an external MCP server."""
    server_name: str
    transport: str  # "stdio" | "sse" | "streamable_http"
    endpoint: Optional[str] = None
    process: Optional[Any] = None  # asyncio.subprocess.Process for stdio
    _session_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MCPClientPool:
    """Pool of external MCP server connections.

    AgentKernel can dynamically discover and call tools from arbitrary MCP
    servers, treating them as *remote skills* governed by the same
    AgentContract budgets that apply to local tools.
    """

    def __init__(self) -> None:
        self._connections: Dict[str, MCPConnection] = {}
        self._remote_tools: Dict[str, MCPConnection] = {}  # tool_name -> conn
        self._lock: asyncio.Lock = asyncio.Lock()

    async def connect_stdio(self, server_name: str, command: List[str]) -> MCPConnection:
        """Spawn an MCP server via stdio and handshake."""
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        conn = MCPConnection(server_name=server_name, transport="stdio", process=proc)
        async with self._lock:
            self._connections[server_name] = conn
        # Simplified handshake: in full implementation, send initialize JSON-RPC
        logger.info("MCP stdio connection established: %s (pid=%s)", server_name, proc.pid)
        return conn

    async def discover_tools(self, server_name: str) -> List[str]:
        """Query an MCP server for available tools and cache them."""
        async with self._lock:
            conn = self._connections.get(server_name)
        if conn is None:
            raise ValueError(f"MCP server {server_name} not connected")
        # Simplified: in full implementation, send tools/list JSON-RPC request
        # For now, return empty list as placeholder for real JSON-RPC
        logger.debug("Tool discovery for %s would happen here (JSON-RPC tools/list)", server_name)
        return []

    async def call_remote_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Call a tool on a remote MCP server."""
        async with self._lock:
            conn = self._connections.get(server_name)
        if conn is None:
            return {"isError": True, "content": [{"type": "text", "text": f"Server {server_name} not connected"}]}
        # Simplified: in full implementation, send tools/call JSON-RPC request
        logger.debug("Remote tool call: %s/%s", server_name, tool_name)
        return {"isError": False, "content": [{"type": "text", "text": "[Remote MCP call placeholder]"}]}

    async def disconnect(self, server_name: str) -> None:
        async with self._lock:
            conn = self._connections.pop(server_name, None)
        if conn and conn.process:
            conn.process.terminate()
            await conn.process.wait()
            logger.info("MCP connection closed: %s", server_name)


# ---------------------------------------------------------------------------
# Capability discovery bridge
# ---------------------------------------------------------------------------

class CapabilityDiscovery:
    """Unified capability directory: local tools + remote MCP tools + A2A agents.

    This provides a single namespace where an Agent can discover *any*
    executable capability, whether it is a local MCP tool, a remote MCP server
    tool, or an A2A agent (see a2a_adapter.py).
    """

    def __init__(
        self,
        local_registry: MCPToolRegistry,
        remote_pool: MCPClientPool,
    ) -> None:
        self._local = local_registry
        self._remote = remote_pool
        self._agent_capabilities: Dict[str, Dict[str, Any]] = {}

    async def register_agent_capability(self, agent_id: _kt.AgentId, card: Dict[str, Any]) -> None:
        """Register an A2A Agent Card into the unified directory."""
        self._agent_capabilities[agent_id.full] = card

    async def discover(self, query: Optional[str] = None) -> Dict[str, Any]:
        """Return all discoverable capabilities matching an optional query."""
        local_tools = await self._local.list_tools()
        return {
            "local_tools": local_tools,
            "remote_servers": list(self._remote._connections.keys()),
            "agents": [
                {"agent_id": aid, "card": card}
                for aid, card in self._agent_capabilities.items()
            ],
        }
