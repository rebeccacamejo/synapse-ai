# a2a_main.py
from __future__ import annotations
import time
import uuid
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# - AgentData model
# - Agent class (optional here)
# - A webhook function or method that actually processes requests:
#     async def webhook(agent_id: str, payload: Dict[str, Any]) -> Dict[str, Any] | str
from agent_main import AgentData
# If you have a module-level webhook, import it:
try:
    from agent_main import webhook as agent_webhook  # type: ignore
except Exception:
    agent_webhook = None  # we'll adapt via a registry if not available

# If you prefer object instances per agent, keep a simple registry.
# Replace with your real agent instances if you have them.
class AgentStub:
    def __init__(self, data: AgentData):
        self.data = data

# ======================
# Models (lean + REST)
# ======================

class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str

class Message(BaseModel):
    role: Literal["user", "agent", "system"]
    parts: List[TextPart]
    messageId: Optional[str] = None
    parentMessageId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SendMessageRequest(BaseModel):
    message: Message
    agentId: Optional[str] = None
    taskId: Optional[str] = None
    contextId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AgentReply(BaseModel):
    message: Message

TaskState = Literal["queued", "running", "completed", "failed", "canceled"]

class Task(BaseModel):
    id: str
    state: TaskState
    createdAt: float
    updatedAt: float
    contextId: Optional[str] = None
    lastMessage: Optional[Message] = None
    error: Optional[str] = None
    agentId: Optional[str] = None

# ---- Agent Card (POC) ----
class AgentCapabilities(BaseModel):
    streaming: bool = False
    pushNotifications: bool = False

class AgentSkill(BaseModel):
    id: str
    name: str
    description: str = ""
    tags: List[str] = []
    inputModes: List[str] = ["text"]
    outputModes: List[str] = ["text"]

class AgentCard(BaseModel):
    protocolVersion: str = "0.3.0"
    name: str
    description: str
    url: str
    version: str
    defaultInputModes: List[str] = ["text"]
    defaultOutputModes: List[str] = ["text"]
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: List[AgentSkill] = Field(default_factory=list)
    supportsAuthenticatedExtendedCard: bool = False

# ======================
# In-memory runtime
# ======================

class TaskStore:
    def __init__(self):
        self._tasks: Dict[str, Task] = {}

    def create(self, *, context_id: Optional[str], agent_id: Optional[str]) -> Task:
        now = time.time()
        tid = str(uuid.uuid4())
        t = Task(id=tid, state="queued", createdAt=now, updatedAt=now,
                 contextId=context_id, agentId=agent_id)
        self._tasks[tid] = t
        return t

    def get(self, task_id: str) -> Task:
        t = self._tasks.get(task_id)
        if not t:
            raise KeyError(task_id)
        return t

    def list(self) -> List[Task]:
        return list(self._tasks.values())

    def update(self, task_id: str, **fields) -> Task:
        t = self.get(task_id)
        for k, v in fields.items():
            setattr(t, k, v)
        t.updatedAt = time.time()
        return t

    def cancel(self, task_id: str) -> Task:
        t = self.get(task_id)
        if t.state not in ("completed", "failed", "canceled"):
            t.state = "canceled"
            t.updatedAt = time.time()
        return t

class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, AgentStub] = {}
        self._cards: Dict[str, AgentCard] = {}
        self._data: Dict[str, AgentData] = {}

    def register(self, agent_id: str, data: AgentData):
        self._data[agent_id] = data
        self._agents[agent_id] = AgentStub(data)
        self._cards[agent_id] = build_agent_card(data)

    def get_card(self, agent_id: Optional[str]) -> AgentCard:
        if agent_id is None:
            if not self._cards:
                raise RuntimeError("No agents registered")
            return self._cards[next(iter(self._cards.keys()))]
        card = self._cards.get(agent_id)
        if not card:
            raise KeyError(agent_id)
        return card

    def get_default_agent_id(self) -> str:
        if not self._agents:
            raise RuntimeError("No agents registered")
        return next(iter(self._agents.keys()))

    def has(self, agent_id: str) -> bool:
        return agent_id in self._agents

# ---- Build an Agent Card from your AgentData ----
def build_agent_card(data: AgentData) -> AgentCard:
    skills: List[AgentSkill] = []
    for s in (data.skills or []):
        if isinstance(s, dict):
            skills.append(
                AgentSkill(
                    id=s.get("id") or s.get("name", "skill"),
                    name=s.get("name") or s.get("id", "skill"),
                    description=s.get("description", ""),
                    tags=s.get("tags", []),
                    inputModes=s.get("input_modes", data.default_input_modes),
                    outputModes=s.get("output_modes", data.default_output_modes),
                )
            )
        else:
            skills.append(AgentSkill(id=str(s), name=str(s)))
    return AgentCard(
        name=data.name,
        description=data.description,
        url=data.url,
        version=data.version,
        defaultInputModes=data.default_input_modes,
        defaultOutputModes=data.default_output_modes,
        capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
        skills=skills,
        supportsAuthenticatedExtendedCard=data.supports_authenticated_extended_card,
    )

# ======================
# Webhook adapter
# ======================
async def call_agent_webhook(agent_id: str, payload: Dict[str, Any]) -> Dict[str, Any] | str:
    """
    Central place to adapt to your actual webhook implementation.

    Expected options you might have in agent_main.py:
      1) Module-level async function:
         async def webhook(agent_id: str, payload: dict) -> dict | str
      2) Class/method on a manager, e.g. Agents().webhook(agent_id, payload)
      3) Per-agent object with .webhook(payload)

    Update this function to call the correct one and normalize the return.
    """
    if agent_webhook:
        return await agent_webhook(agent_id, payload)  # type: ignore

    # Fallback example: raise if not wired
    raise RuntimeError("No webhook callable wired. Please wire agent_main.webhook(...) here.")

# ======================
# App setup
# ======================

app = FastAPI(title="A2A REST POC (no SDK, no SSE)")

registry = AgentRegistry()

# ---- Register at least one agent (adjust to your config) ----
default_agent_id = "default"
default_agent_data = AgentData(
    name="Enterprise n8n-like Agent",
    description="POC A2A REST server (no SDK, no SSE).",
    version="0.1.0",
    url="http://localhost:8000/",
    default_input_modes=["text"],
    default_output_modes=["text"],
    skills=[{"id": "chat", "name": "General Chat"}],
    supports_authenticated_extended_card=False,
    streaming=False,
)
registry.register(default_agent_id, default_agent_data)

task_store = TaskStore()

# ======================
# Endpoints
# ======================

# 1) Agent Card at the well-known path (public)
@app.get("/.well-known/agent-card.json")
async def get_agent_card():
    # Default card (first registered)
    try:
        card = registry.get_card(None)
        return JSONResponse(card.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2) v1/getcard (optionally per agentId; this is what you asked to add)
@app.get("/v1/getcard")
async def get_card_v1(agentId: Optional[str] = Query(default=None, alias="agentId")):
    """
    Returns the Agent Card JSON. If agentId is provided, returns that agent’s card.
    Example:
      GET /v1/getcard
      GET /v1/getcard?agentId=billing-bot
    """
    try:
        card = registry.get_card(agentId)
        return JSONResponse(card.model_dump())
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown agentId: {agentId}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3) Send a message (non-streaming) — delegates to your webhook/{agent_id}
@app.post("/v1/message:send")
async def message_send(req: SendMessageRequest):
    agent_id = req.agentId or registry.get_default_agent_id()
    if not registry.has(agent_id):
        raise HTTPException(status_code=404, detail=f"unknown agentId: {agent_id}")

    # Create a task (handy even without streaming)
    task = task_store.create(context_id=req.contextId, agent_id=agent_id)
    task_store.update(task.id, state="running")

    # Build a minimal payload for your webhook
    user_text = " ".join([p.text for p in req.message.parts if p.kind == "text"]).strip()
    payload: Dict[str, Any] = {
        "taskId": task.id,
        "contextId": req.contextId,
        "metadata": req.metadata or {},
        "message": {
            "role": req.message.role,
            "parts": [p.model_dump() for p in req.message.parts],
            "messageId": req.message.messageId,
            "parentMessageId": req.message.parentMessageId,
        },
        "text": user_text,  # convenience field most webhook handlers expect
    }

    try:
        # >>> Delegate to your real logic
        result = await call_agent_webhook(agent_id, payload)

        # Normalize webhook result → final agent text
        if isinstance(result, str):
            result_text = result
        elif isinstance(result, dict):
            # Common shapes: {"text": "..."} or {"message": {"parts":[{"kind":"text","text":"..."}]}}
            if "text" in result and isinstance(result["text"], str):
                result_text = result["text"]
            elif "message" in result and isinstance(result["message"], dict):
                parts = result["message"].get("parts", [])
                # find first text part
                result_text = ""
                for part in parts:
                    if isinstance(part, dict) and part.get("kind") == "text":
                        result_text = str(part.get("text", ""))
                        if result_text:
                            break
            else:
                result_text = str(result)
        else:
            result_text = str(result)

        agent_msg = Message(
            role="agent",
            parts=[TextPart(text=result_text)],
            parentMessageId=req.message.messageId,
        )
        task = task_store.update(task.id, state="completed", lastMessage=agent_msg)
        return JSONResponse({"reply": AgentReply(message=agent_msg).model_dump(),
                             "task": task.model_dump()})
    except Exception as e:
        task = task_store.update(task.id, state="failed", error=str(e))
        raise HTTPException(status_code=500, detail={"task": task.model_dump()})

# 4) Tasks list/get/cancel
@app.get("/v1/tasks")
async def tasks_list():
    return JSONResponse({"tasks": [t.model_dump() for t in task_store.list()]})

@app.get("/v1/tasks/{task_id}")
async def tasks_get(task_id: str):
    try:
        return JSONResponse(task_store.get(task_id).model_dump())
    except KeyError:
        raise HTTPException(status_code=404, detail="task not found")

@app.post("/v1/tasks/{task_id}:cancel")
async def tasks_cancel(task_id: str):
    try:
        t = task_store.cancel(task_id)
        # If your webhook supports cooperative cancellation, signal it here.
        return JSONResponse(t.model_dump())
    except KeyError:
        raise HTTPException(status_code=404, detail="task not found")

# ------------- Run -------------
# uvicorn a2a_main:app --reload --port 8000
