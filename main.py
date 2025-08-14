# a2a_main.py
# -----------------------------------------------------------------------------
# Standalone FastAPI app implementing the A2A REST endpoints.
# - Public Agent Card:        GET /.well-known/agent-card.json
# - Auth'd Agent Card:        GET /v1/card
# - Send message (blocking):  POST /v1/message:send
# - Send message (stream):    POST /v1/message:stream  (SSE)
# - Tasks:                    GET /v1/tasks
#                             GET /v1/tasks/{task_id}
#                             POST /v1/tasks/{task_id}:cancel
#                             POST /v1/tasks/{task_id}:subscribe (SSE)
# - Push notification cfgs:   POST/GET/GET/DELETE /v1/tasks/{task_id}/pushNotificationConfigs
#
# Storage here is in-memory for clarity; swap for Redis/Postgres in prod.
# It *invokes your existing* /webhook/{agent_id} endpoint to execute each turn.
# -----------------------------------------------------------------------------

from typing import Dict, List, Optional, Any, AsyncGenerator
from uuid import uuid4
from datetime import datetime, timezone
import os

from fastapi import FastAPI, Body, Header, HTTPException, Path, Query
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

# -----------------------------------------------------------------------------
# AgentData — describes your agent and powers the Agent Card.
# You said this already exists; keep this here as a compatible shape.
# -----------------------------------------------------------------------------
class AgentData:
    def __init__(
        self,
        agent_id: str,
        name: str,
        description: str,
        version: str,
        base_url: str,
        system_prompt: str,
        mcp_servers: List[dict],
        llm_models: List[dict],
        security: dict = None,
        capabilities: dict = None,
        icon_url: Optional[str] = None,
        docs_url: Optional[str] = None,
        default_input_modes: Optional[List[str]] = None,
        default_output_modes: Optional[List[str]] = None,
        skills: Optional[List[dict]] = None,
        additional_interfaces: Optional[List[dict]] = None,
        supports_authenticated_extended_card: bool = True,
        protocol_version: str = "1.0",
        preferred_transport: str = "HTTP",
        provider: Optional[dict] = None,
    ):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.version = version
        self.base_url = base_url  # Public base URL of THIS service
        self.system_prompt = system_prompt
        self.mcp_servers = mcp_servers
        self.llm_models = llm_models
        self.security = security or {}
        self.capabilities = capabilities or {}
        self.icon_url = icon_url
        self.docs_url = docs_url
        self.default_input_modes = default_input_modes or ["text/plain"]
        self.default_output_modes = default_output_modes or ["text/plain"]
        self.skills = skills or []
        self.additional_interfaces = additional_interfaces or []
        self.supports_authenticated_extended_card = supports_authenticated_extended_card
        self.protocol_version = protocol_version
        self.preferred_transport = preferred_transport
        self.provider = provider or {"organization": "Your Org", "url": self.base_url}

# -----------------------------------------------------------------------------
# Initialize app
# -----------------------------------------------------------------------------
app = FastAPI(title="A2A FastAPI Server", version="1.0.0")

# -----------------------------------------------------------------------------
# In-memory stores (replace with persistence in production)
# -----------------------------------------------------------------------------
TASKS: Dict[str, Dict[str, Any]] = {}
MESSAGES: Dict[str, List[Dict[str, Any]]] = {}
PUSH_CONFIGS: Dict[str, Dict[str, Any]] = {}

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def make_task(task_id: str, state: str, history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a minimal Task object with status + history."""
    return {
        "id": task_id,
        "status": {"state": state, "updatedAt": now_iso()},
        "artifacts": [],
        "history": history,
        "metadata": {},
    }

def make_message(role: str, text: str) -> Dict[str, Any]:
    """Minimal A2A Message object (role + text part + timestamp)."""
    return {
        "role": role,  # "user" | "agent" | "system"
        "parts": [{"kind": "text", "text": text}],
        "timestamp": now_iso(),
    }

# -----------------------------------------------------------------------------
# Auth — enforce what you declare in the Agent Card (securitySchemes/security).
# Replace with real API key/JWT checks for your environment.
# -----------------------------------------------------------------------------
def require_auth(x_api_key: Optional[str], authorization: Optional[str]) -> None:
    if x_api_key:
        return
    if authorization and authorization.lower().startswith("bearer "):
        # TODO: Validate JWT or your token
        return
    raise HTTPException(status_code=401, detail="Unauthorized")

# -----------------------------------------------------------------------------
# Agent Card builders (from AgentData)
# -----------------------------------------------------------------------------
def build_public_card(agent: AgentData) -> Dict[str, Any]:
    """Public discovery Agent Card (recommended at /.well-known/agent-card.json)."""
    return {
        "protocolVersion": agent.protocol_version,
        "version": agent.version,
        "name": agent.name,
        "description": agent.description,
        "provider": agent.provider,
        "url": agent.base_url,
        "preferredTransport": agent.preferred_transport,
        "iconUrl": agent.icon_url,
        "documentationUrl": agent.docs_url,
        "capabilities": {
            "streaming": bool(agent.capabilities.get("streaming", True)),
            "pushNotifications": bool(agent.capabilities.get("pushNotifications", True)),
            "stateTransitionHistory": bool(agent.capabilities.get("stateTransitionHistory", True)),
            "extensions": agent.capabilities.get("extensions", []),
        },
        "defaultInputModes": agent.default_input_modes,
        "defaultOutputModes": agent.default_output_modes,
        "securitySchemes": agent.security.get("schemes", {}),
        "security": agent.security.get("security", []),  # e.g. [["ApiKey"], ["Bearer"]]
        "skills": agent.skills,
        "additionalInterfaces": agent.additional_interfaces,
        "supportsAuthenticatedExtendedCard": agent.supports_authenticated_extended_card,
        "signatures": agent.security.get("signatures", []),
    }

def build_authenticated_card(agent: AgentData) -> Dict[str, Any]:
    """Authenticated/extended Agent Card (may include extra skills/config)."""
    card = build_public_card(agent)
    extra_skills = [
        {
            "id": "invoke-mcp",
            "name": "Invoke MCP Tool",
            "description": "Curated MCP tool invocations behind the agent.",
            "inputModes": ["application/json"],
            "outputModes": ["application/json"],
            "tags": ["mcp", "llm"],
        }
    ]
    card["skills"] = card.get("skills", []) + extra_skills
    return card

# -----------------------------------------------------------------------------
# Load AgentData (envs shown so you can override at deploy time)
# -----------------------------------------------------------------------------
AGENT = AgentData(
    agent_id=os.getenv("AGENT_ID", "default"),
    name=os.getenv("AGENT_NAME", "Example A2A Agent"),
    description=os.getenv("AGENT_DESCRIPTION", "A2A-compliant agent backed by MCP + LLM"),
    version=os.getenv("AGENT_VERSION", "1.0.0"),
    base_url=os.getenv("PUBLIC_BASE_URL", "http://localhost:8000"),
    system_prompt=os.getenv("SYSTEM_PROMPT", "You are a helpful agent..."),
    mcp_servers=[{"name": "files", "url": os.getenv("MCP_FILES_URL", "http://mcp-files:3000")}],
    llm_models=[{"provider": os.getenv("LLM_PROVIDER", "openai"), "model": os.getenv("LLM_MODEL", "gpt-4o")}],
    security={
        "schemes": {
            "ApiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"},
            "Bearer": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"},
        },
        "security": [["ApiKey"], ["Bearer"]],
    },
)

# -----------------------------------------------------------------------------
# AGENT CARD ENDPOINTS
# -----------------------------------------------------------------------------
@app.get("/.well-known/agent-card.json")
def get_public_card():
    """
    Public discovery endpoint.
    Lets clients discover transport + auth + skills without authentication.
    """
    return JSONResponse(build_public_card(AGENT))

@app.get("/v1/card")
def get_authenticated_card(
    x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)
):
    """
    Authenticated Agent Card (may include extra skills/config compared to public).
    """
    require_auth(x_api_key, authorization)
    return JSONResponse(build_authenticated_card(AGENT))

# -----------------------------------------------------------------------------
# MESSAGE SEND (non-streaming)
# -----------------------------------------------------------------------------
@app.post("/v1/message:send")
async def message_send(
    payload: Dict[str, Any] = Body(...),
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """
    Handle a single non-streaming message turn and return { task }:
      - Accepts: { message, configuration?, metadata? }
      - Appends incoming message to the task's history
      - Calls your existing /webhook/{agent_id} to execute MCP+LLM turn
      - Appends agent reply to history
      - Returns the Task (id, status, entire history)
    """
    require_auth(x_api_key, authorization)

    msg = payload.get("message")
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")
    config = payload.get("configuration", {}) or {}

    # Create or reuse taskId (lets clients have multi-turn tasks)
    task_id = msg.get("taskId") or str(uuid4())

    # Persist conversation history for this task
    history = MESSAGES.setdefault(task_id, [])
    history.append(msg)

    # Invoke your existing webhook that runs the agent (MCP + LLM)
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{AGENT.base_url}/webhook/{AGENT.agent_id}",
            json={
                "taskId": task_id,
                "history": history,
                "config": config,
                "agentData": {
                    "system_prompt": AGENT.system_prompt,
                    "mcp_servers": AGENT.mcp_servers,
                    "llm_models": AGENT.llm_models,
                },
            },
            headers={"X-API-Key": x_api_key} if x_api_key else {},
        )
    if resp.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"webhook error: {resp.text}")

    result = resp.json()
    reply_text = result.get("replyText", "Acknowledged.")  # adapt to your webhook's schema

    # Record the agent's reply and finalize state
    agent_reply = make_message("agent", reply_text)
    history.append(agent_reply)
    state = "COMPLETED" if config.get("blocking", True) else "RUNNING"

    TASKS[task_id] = make_task(task_id, state, history)
    return {"task": TASKS[task_id]}

# -----------------------------------------------------------------------------
# MESSAGE STREAM (SSE)
# -----------------------------------------------------------------------------
@app.post("/v1/message:stream")
async def message_stream(
    payload: Dict[str, Any] = Body(...),
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """
    Streaming variant using Server-Sent Events (SSE).
    If your /webhook supports SSE/chunking, bridge its events here.
    For demo, we fetch a full reply and emit one delta before completing.
    """
    require_auth(x_api_key, authorization)

    msg = payload.get("message")
    if not msg:
        raise HTTPException(status_code=400, detail="message is required")

    task_id = msg.get("taskId") or str(uuid4())

    # Start or extend history, mark task running
    history = MESSAGES.setdefault(task_id, [])
    history.append(msg)
    TASKS[task_id] = make_task(task_id, "RUNNING", history)

    async def eventgen() -> AsyncGenerator[bytes, None]:
        # 1) Tell client we started
        yield f"data={{\"event\":\"status\",\"taskId\":\"{task_id}\",\"state\":\"RUNNING\"}}\n\n".encode()

        # 2) Call your webhook (non-streaming example)
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{AGENT.base_url}/webhook/{AGENT.agent_id}",
                json={"taskId": task_id, "history": history},
                headers={"X-API-Key": x_api_key} if x_api_key else {},
            )
        resp.raise_for_status()
        reply_text = resp.json().get("replyText", "Stream complete.")

        # 3) Emit a small delta (replace with true token streaming if available)
        yield f"data={{\"event\":\"message\",\"taskId\":\"{task_id}\",\"delta\":{reply_text[:80]!r}}}\n\n".encode()

        # 4) Finalize the task and notify completion
        agent_reply = make_message("agent", reply_text)
        history.append(agent_reply)
        TASKS[task_id]["status"]["state"] = "COMPLETED"
        TASKS[task_id]["status"]["updatedAt"] = now_iso()
        yield f"data={{\"event\":\"status\",\"taskId\":\"{task_id}\",\"state\":\"COMPLETED\"}}\n\n".encode()

    return StreamingResponse(eventgen(), media_type="text/event-stream")

# -----------------------------------------------------------------------------
# TASKS API
# -----------------------------------------------------------------------------
@app.get("/v1/tasks")
def list_tasks(
    x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)
):
    """List all tasks (add filtering/pagination/tenancy in prod)."""
    require_auth(x_api_key, authorization)
    return list(TASKS.values())

@app.get("/v1/tasks/{task_id}")
def get_task(
    task_id: str = Path(...),
    historyLength: Optional[int] = Query(None),
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """
    Get a task by id. Optional ?historyLength=N returns only the last N messages,
    which helps UIs render recent context without pulling the entire log.
    """
    require_auth(x_api_key, authorization)
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if historyLength is not None:
        return {**task, "history": task.get("history", [])[-historyLength:]}
    return task

@app.post("/v1/tasks/{task_id}:cancel")
def cancel_task(
    task_id: str,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """
    Best-effort cancel. Wire this to your worker’s cancellation signal so long
    tasks can actually stop rather than just flipping state here.
    """
    require_auth(x_api_key, authorization)
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task["status"]["state"] = "CANCELED"
    task["status"]["updatedAt"] = now_iso()
    return task

@app.post("/v1/tasks/{task_id}:subscribe")
def resubscribe_task(
    task_id: str,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """
    Reconnect to an existing task via SSE (e.g., after the UI reconnects).
    Real impls may backfill missed events or continue live if still running.
    """
    require_auth(x_api_key, authorization)
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")

    async def eventgen() -> AsyncGenerator[bytes, None]:
        yield f"data={{\"event\":\"resubscribed\",\"taskId\":\"{task_id}\"}}\n\n".encode()

    return StreamingResponse(eventgen(), media_type="text/event-stream")

# -----------------------------------------------------------------------------
# PUSH NOTIFICATION CONFIGS
# -----------------------------------------------------------------------------
@app.post("/v1/tasks/{task_id}/pushNotificationConfigs")
def set_push_config(
    task_id: str,
    payload: Dict[str, Any] = Body(...),
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """
    Create/update a push notification config for a task.
    Expected payload: { "config": { ... } } (server assigns id if missing).
    """
    require_auth(x_api_key, authorization)
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")

    cfg = payload.get("config")
    if not cfg:
        raise HTTPException(status_code=400, detail="config is required")

    cfg_id = cfg.get("id") or str(uuid4())
    cfg["id"] = cfg_id
    cfg["createdAt"] = now_iso()
    PUSH_CONFIGS.setdefault(task_id, {})[cfg_id] = cfg
    return cfg

@app.get("/v1/tasks/{task_id}/pushNotificationConfigs")
def list_push_configs(
    task_id: str,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """List all push notification configs registered for this task."""
    require_auth(x_api_key, authorization)
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="Task not found")
    return list(PUSH_CONFIGS.get(task_id, {}).values())

@app.get("/v1/tasks/{task_id}/pushNotificationConfigs/{config_id}")
def get_push_config(
    task_id: str,
    config_id: str,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Retrieve a specific push notification config by id."""
    require_auth(x_api_key, authorization)
    cfg = PUSH_CONFIGS.get(task_id, {}).get(config_id)
    if not cfg:
        raise HTTPException(status_code=404, detail="Push notification config not found")
    return cfg

@app.delete("/v1/tasks/{task_id}/pushNotificationConfigs/{config_id}")
def delete_push_config(
    task_id: str,
    config_id: str,
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """Delete a push notification config and return the removed object."""
    require_auth(x_api_key, authorization)
    store = PUSH_CONFIGS.get(task_id, {})
    if config_id not in store:
        raise HTTPException(status_code=404, detail="Push notification config not found")
    return store.pop(config_id)
