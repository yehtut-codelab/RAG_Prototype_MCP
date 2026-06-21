import json
import logging
import time
import uuid
from contextlib import asynccontextmanager, AsyncExitStack
from typing import AsyncIterator, Callable, Awaitable

logger = logging.getLogger(__name__)

import truststore
truststore.inject_into_ssl()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import settings
from models import ChatCompletionRequest, ModelInfo, ModelsResponse
from rag_graph import create_rag_agent

# Schema is loaded once at startup and reused by every request.
_db_schema: str | None = None


class _StripSemicolonInterceptor:
    """Strip trailing semicolons before SQL reaches mcp-postgres.

    mcp-postgres treats any query ending with ';' as multi-statement and
    rejects it. LLMs reliably append semicolons, so fix it here in code
    rather than relying on prompt instructions.
    MCPToolCallRequest uses .args (not .arguments).
    """
    async def __call__(self, request, handler):
        if "query" in request.args:
            request.args["query"] = (
                request.args["query"].strip().rstrip(";").strip()
            )
        return await handler(request)


async def _build_agent(db_schema: str | None, stack: AsyncExitStack):
    """Spin up a fresh MCP session and return a ready agent.

    The session (and its npx subprocess) lives for the lifetime of *stack*.
    Callers must call stack.aclose() when the request is done.
    """
    mcp_tools = None

    if settings.mcp_enabled:
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
            from langchain_mcp_adapters.tools import load_mcp_tools

            client = MultiServerMCPClient(
                {
                    "postgres": {
                        "command": "npx",
                        "args": ["-y", "mcp-postgres"],
                        "env": {"DATABASE_URL": settings.postgres_url},
                        "transport": "stdio",
                    }
                },
                tool_interceptors=[_StripSemicolonInterceptor()],
            )
            session = await stack.enter_async_context(client.session("postgres"))
            mcp_tools = await load_mcp_tools(
                session,
                tool_interceptors=[_StripSemicolonInterceptor()],
            )
        except Exception as e:
            logger.error("MCP session failed: %s", e)
            mcp_tools = None

    return create_rag_agent(mcp_tools, db_schema)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db_schema

    try:
        from schema_loader import load_schema
        _db_schema = load_schema(settings.postgres_url)
        print("Database schema loaded.")
    except Exception as e:
        print(f"Schema preload failed ({e}), agent will discover schema via tools.")

    print("Server ready.")
    yield


app = FastAPI(title="RAG MCP API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.llm_model}


@app.get("/v1/models")
async def list_models():
    return ModelsResponse(
        data=[ModelInfo(id="rag-model", created=int(time.time()))]
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:
        return StreamingResponse(
            _stream_sse(messages, request.model),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    stack = AsyncExitStack()
    try:
        agent = await _build_agent(_db_schema, stack)
        result = await agent.ainvoke(
            {"messages": messages},
            config={"recursion_limit": 50},
        )
        content = result["messages"][-1].content
    except Exception as e:
        logger.error("Agent invocation failed: %s", e, exc_info=True)
        content = "I'm sorry, I encountered an issue processing your request. Please try again."
    finally:
        await stack.aclose()

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _stream_sse(messages: list, model: str) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    def make_chunk(text: str, finish: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text} if text else {},
                    "finish_reason": finish,
                }
            ],
        }
        return f"data: {json.dumps(payload)}\n\n"

    stack = AsyncExitStack()
    try:
        agent = await _build_agent(_db_schema, stack)

        async for event in agent.astream_events(
            {"messages": messages},
            version="v2",
            config={"recursion_limit": 50},
        ):
            if event["event"] != "on_chat_model_stream":
                continue

            chunk = event["data"].get("chunk")
            if chunk is None:
                continue

            content = chunk.content
            text = ""

            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += block.get("text", "")
                    elif isinstance(block, str):
                        text += block

            if text:
                yield make_chunk(text)

    except Exception as e:
        logger.error("Streaming agent error: %s", e, exc_info=True)
        yield make_chunk("I'm sorry, I encountered an issue processing your request. Please try again.")
    finally:
        await stack.aclose()

    yield make_chunk("", finish="stop")
    yield "data: [DONE]\n\n"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True,
    )
