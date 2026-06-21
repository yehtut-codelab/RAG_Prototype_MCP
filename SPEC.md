# RAG + MCP Prototype — Technical Specification

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Architecture](#3-architecture)
4. [Component Deep-Dives](#4-component-deep-dives)
   - 4.1 [FastAPI Backend](#41-fastapi-backend-mainpy)
   - 4.2 [LangGraph Agent](#42-langgraph-agent-rag_graphpy)
   - 4.3 [MCP Integration](#43-mcp-integration)
   - 4.4 [Schema Preloading](#44-schema-preloading-schema_loaderpy)
   - 4.5 [Vector Store](#45-vector-store-vector_storepy)
   - 4.6 [Configuration](#46-configuration-configpy)
   - 4.7 [Chainlit UI](#47-chainlit-ui-chainlit_appappy)
5. [MCP Architecture — Detailed](#5-mcp-architecture--detailed)
   - 5.1 [Transport Model: stdio vs HTTP](#51-transport-model-stdio-vs-http)
   - 5.2 [Per-Request Session Architecture](#52-per-request-session-architecture)
   - 5.3 [The 17 MCP Tools](#53-the-17-mcp-tools)
   - 5.4 [Internal vs External MCP Server](#54-internal-vs-external-mcp-server)
   - 5.5 [Adding More Tools](#55-adding-more-tools)
6. [LLM Intelligence — How Reports Work Without Custom Code](#6-llm-intelligence--how-reports-work-without-custom-code)
6. [Data Flow — Query Lifecycle](#6-data-flow--query-lifecycle)
7. [Key Technical Decisions & Rationale](#7-key-technical-decisions--rationale)
8. [Known Issues & Bugs Fixed](#8-known-issues--bugs-fixed)
9. [Configuration Reference](#9-configuration-reference)
10. [Package Versions](#10-package-versions)
11. [Running the Services](#11-running-the-services)
12. [Database — Chinook Schema](#12-database--chinook-schema)

---

## 1. Project Overview

A prototype that combines two retrieval strategies behind a single chat API:

| Strategy | What it does |
|---|---|
| **RAG (vector search)** | Embeds documents and searches by semantic similarity using PGVector |
| **MCP (database tools)** | Gives an LLM agent direct SQL access to PostgreSQL via structured tool calls |

The agent decides at runtime which strategy to use based on the question. Document questions go to vector search; structured data questions go to SQL via MCP tools. The API is OpenAI-compatible so any client that speaks `POST /v1/chat/completions` can use it.

---

## 2. Directory Structure

```
RAG_Prototype_MCP/
├── .env                        # Secrets and runtime config (not committed)
├── .venv/                      # Python virtual environment
├── start_service.cmd           # Launch backend on port 8000
├── start_chainlit.cmd          # Launch Chainlit UI on port 8001
├── ARCHITECTURE.md             # Query flow diagram and tool list
├── SPEC.md                     # This file
│
├── backend/
│   ├── main.py                 # FastAPI app, lifespan, endpoints
│   ├── rag_graph.py            # LangGraph agent definition
│   ├── vector_store.py         # PGVector wrapper (lazy init)
│   ├── schema_loader.py        # Sync PostgreSQL schema reader
│   ├── config.py               # pydantic-settings config
│   └── models.py               # Pydantic request/response models
│
└── chainlit_app/
    └── app.py                  # Chainlit chat UI
```

---

## 3. Architecture

```
User (browser)
    │
    ▼
Chainlit UI  (port 8001)
    │  AsyncOpenAI client → POST /v1/chat/completions  (stream=True)
    ▼
FastAPI Backend  (port 8000)
    │
    ├── StreamingResponse  (SSE / text-event-stream)
    │       │
    │       ▼
    │   LangGraph ReAct Agent
    │       │
    │       ├── Tool: search_documents
    │       │       └── PGVector similarity_search()
    │       │               └── PostgreSQL  (langchain_postgres collection)
    │       │
    │       └── Tools: 17 MCP tools  (execute_raw_query, query_data, …)
    │               └── mcp-postgres subprocess  (stdio pipe, persistent)
    │                       └── PostgreSQL  (direct SQL)
    │
    └── OpenAI API  (gpt-4o-mini)
```

### Startup sequence

1. `lifespan()` runs once when FastAPI starts.
2. Spawns `npx mcp-postgres` as a child process (stdio), enters persistent session.
3. Calls `load_mcp_tools(session)` — loads 17 tool definitions from the subprocess.
4. Calls `load_schema()` — synchronous psycopg query to `information_schema`; result is ~720 tokens of schema text.
5. Builds `create_rag_agent(mcp_tools, db_schema)` — configures LangGraph with tools and schema-aware system prompt.
6. Server begins accepting requests.

---

## 4. Component Deep-Dives

### 4.1 FastAPI Backend (`main.py`)

**Endpoints**

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns `{"status":"ok","model":"gpt-4o-mini"}` |
| GET | `/v1/models` | Returns model list (OpenAI-compatible) |
| POST | `/v1/chat/completions` | Main chat endpoint; supports `stream: true/false` |

**Streaming (`stream: true`)**

Uses `StreamingResponse` with `media_type="text/event-stream"`. The async generator `_stream_sse()` drives `rag_agent.astream_events(version="v2")` and filters for `on_chat_model_stream` events only. Tool call events are silently consumed — the user never sees intermediate tool output. Each token is formatted as an SSE `data:` line with an OpenAI-compatible JSON chunk.

**Non-streaming (`stream: false`)**

Calls `rag_agent.ainvoke()` and returns the last message from the result. Suitable for programmatic callers but times out on complex queries in PowerShell's default HTTP client (~60–90 s). Streaming is the recommended path for the UI.

**Error handling**

All exceptions inside the agent are caught and logged server-side via `logger.error(..., exc_info=True)`. The user receives a generic retry message. Internal errors, SQL failures, and recursion limit errors are never exposed to the client.

**SSL on Windows**

```python
import truststore
truststore.inject_into_ssl()
```

Called at module top before any HTTPS requests. Injects Windows certificate store into Python's SSL context so that calls to `api.openai.com` succeed without `SSLCertVerificationError`.

**CORS**

Wide-open (`allow_origins=["*"]`) for prototype use. Restrict to specific origins before production.

**Lifespan and MCP cleanup**

The MCP subprocess is managed through `AsyncExitStack`. On shutdown (SIGTERM or Ctrl-C), `_mcp_stack.aclose()` is called in the `finally` block, which closes the stdio pipe and terminates the child process cleanly.

---

### 4.2 LangGraph Agent (`rag_graph.py`)

**Framework:** `langgraph.prebuilt.create_react_agent`

The agent follows the ReAct pattern: it alternates between reasoning (LLM call) and acting (tool call) until it produces a final answer. Each think → tool call → observe cycle counts as steps in the graph.

**Recursion limit:** 50

With schema preloading, simple queries need 3 steps (think → 1 tool call → respond). Complex BA queries with multiple joins use at most 5–7 steps. 50 is a safe ceiling.

**System prompt construction**

The prompt is built dynamically in `create_rag_agent()`:

```
[Base prompt with guidelines]
  ↓
IF schema is preloaded:
    "The database schema is already provided — skip list_tables/describe_table,
     go directly to query_data or execute_raw_query"
ELSE:
    "Call list_tables then describe_table to understand schema first"
  ↓
IF schema is preloaded:
    [Full schema text appended: TABLE: album, TABLE: artist, ...]
```

This avoids ~4–6 redundant schema-discovery tool calls per query, which was the root cause of `GraphRecursionError` before schema preloading.

**Behavioral guardrails in system prompt**

- Never supplement database results with training knowledge
- If DB returns fewer results than the user expects, report actual results and explain
- One SQL statement per tool call — no semicolon-chained statements
- If a query fails, try once simpler. If it fails again, stop and report

**LLM:** `ChatOpenAI(model="gpt-4o-mini")`

Switched from Anthropic Claude (insufficient credits on the account used for this prototype). OpenAI `gpt-4o-mini` is cost-effective and handles multi-tool ReAct loops well.

---

### 4.3 MCP Integration

#### What is MCP?

Model Context Protocol (MCP) is a standard for exposing tools, resources, and prompts to LLM clients. A server advertises its capabilities; a client (the agent) discovers and calls them. This prototype uses it to give the agent structured SQL access to PostgreSQL.

#### Package used

`langchain-mcp-adapters 0.3.0` — bridges MCP tool definitions into LangChain/LangGraph `BaseTool` instances.

#### MCP server

`mcp-postgres` (npm package). Installed globally via npx. Exposes 17 tools over stdio transport.

#### Transport: stdio

The backend spawns `npx mcp-postgres` as a **child process** and communicates via stdin/stdout pipes. This is the stdio transport model:

```
FastAPI process
  └── subprocess: npx mcp-postgres
        stdin  ←── tool call JSON
        stdout ──► tool result JSON
        └── connects to PostgreSQL
```

**Key characteristic:** stdio requires a parent-child relationship. You cannot connect to an already-running stdio MCP server from outside — there is no network socket.

#### Persistent session (critical)

`langchain-mcp-adapters 0.3.0` supports two usage modes:

| Mode | API | Behaviour |
|---|---|---|
| Per-call (default) | `await client.get_tools()` | Spawns a NEW `npx` subprocess for EVERY tool invocation |
| Persistent session | `async with client.session("postgres") as s` | One subprocess for the entire session |

**The prototype uses per-request sessions.** Each HTTP request spawns one `npx mcp-postgres` subprocess, all tool calls within that request share it, and the subprocess is shut down when the request ends.

This went through three iterations before reaching the current design:

#### Iteration 1 — Per-call (broken)

Default `get_tools()` mode. A new subprocess spawned for every single tool invocation. A complex BA query calling 16 tools launched 16 processes → `GraphRecursionError`.

#### Iteration 2 — Server-lifetime persistent session (fragile)

One subprocess shared across ALL requests for the server's entire lifetime. Fast, but the subprocess could die silently while `/health` continued returning `ok`. All subsequent queries then failed with the LLM's "I am unable to retrieve" message, with no error logged at the API level.

#### Iteration 3 — Per-request session (current, stable)

One subprocess per HTTP request. Fresh on every call, closed in a `finally` block. Mimics HTTP's stateless-per-request model within stdio's constraints.

```python
async def _build_agent(db_schema, stack: AsyncExitStack):
    client = MultiServerMCPClient(
        {"postgres": {"command": "npx", "args": ["-y", "mcp-postgres"], ...}},
        tool_interceptors=[_StripSemicolonInterceptor()],
    )
    session = await stack.enter_async_context(client.session("postgres"))
    mcp_tools = await load_mcp_tools(
        session,
        tool_interceptors=[_StripSemicolonInterceptor()],  # must pass here too
    )
    return create_rag_agent(mcp_tools, db_schema)

# In request handler and streaming generator:
stack = AsyncExitStack()
try:
    agent = await _build_agent(_db_schema, stack)
    # ... run agent ...
finally:
    await stack.aclose()  # kills subprocess cleanly
```

**Why `tool_interceptors` must be passed to both `MultiServerMCPClient` and `load_mcp_tools`:**
`MultiServerMCPClient`'s `tool_interceptors` parameter only applies when using `get_tools()`. When using `client.session()` + `load_mcp_tools(session)`, interceptors must be passed directly to `load_mcp_tools()`. Passing only to the client constructor has no effect in session mode.

**Tradeoff:** One `npx` subprocess startup (~0.5–1 s) per request. Acceptable for a chat interface where agent reasoning already takes several seconds. Would not be acceptable for a high-throughput API.

Note: `MultiServerMCPClient.__aenter__` deliberately raises `NotImplementedError` in v0.1.0+. Do not use it as `async with MultiServerMCPClient(...)`. Use `client.session()` instead.

---

### 4.4 Schema Preloading (`schema_loader.py`)

At startup, `load_schema()` queries `information_schema` and builds a compact schema string (~720 tokens) that is injected into the agent's system prompt.

**Why synchronous psycopg, not async?**

Python on Windows uses `ProactorEventLoop` (IOCP). `psycopg.AsyncConnection` is incompatible with ProactorEventLoop. The synchronous `psycopg.connect()` call in `lifespan()` blocks the event loop briefly at startup only — acceptable because no requests are being served yet.

**Schema format**

```
DATABASE SCHEMA
========================================

TABLE: album
  - album_id (integer)  [PK, NOT NULL]
  - title (character varying)  [NOT NULL]
  - artist_id (integer)  [FK → artist.artist_id, NOT NULL]

TABLE: artist
  ...
========================================
```

Includes: column names, data types, nullability, primary keys, foreign key references with target table and column.

**Effect**

Before schema preloading: agent called `list_tables` → `describe_table` (×N) before every query = 4–8 extra tool calls.
After schema preloading: agent reads schema from system prompt and calls `execute_raw_query` directly = 1 tool call.

---

### 4.5 Vector Store (`vector_store.py`)

**Engine:** `langchain-postgres PGVector` — stores document embeddings in a PostgreSQL `vector` column.

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, runs on CPU, 384-dimensional vectors)

**Lazy initialization:** The embedding model (~90 MB) is only loaded on the first `similarity_search()` call, not at startup. This keeps startup time fast.

**Collection:** Configured via `COLLECTION_NAME` env var (default `rag_documents`). Stored in a `langchain_pg_collection` + `langchain_pg_embedding` table pair created automatically by PGVector.

**SQLAlchemy URL:** PGVector requires `postgresql+psycopg://` prefix (SQLAlchemy dialect). `config.py` exposes `postgres_sqlalchemy_url` which converts from the base `postgresql://` URL automatically.

**Usage by agent:** The `search_documents` tool calls `vs.similarity_search(query, k=5)`. If the vector store is unavailable (e.g., pgvector extension not installed), it catches the exception and returns a graceful fallback string, so the agent can pivot to MCP tools instead.

---

### 4.6 Configuration (`config.py`)

Uses `pydantic-settings`. Reads `.env` from the project root (one level above `backend/`).

**Path resolution fix:** `.env` path is resolved relative to `config.py`'s location (`Path(__file__).parent.parent / ".env"`), not the working directory. This ensures the file is found regardless of which directory the server is launched from.

```python
_ENV_FILE = Path(__file__).parent.parent / ".env"
```

**All settings**

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_URL` | `postgresql://postgres:Password123@localhost:5432/postgres` | PostgreSQL connection string |
| `OPENAI_API_KEY` | `""` | OpenAI API key (required) |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `COLLECTION_NAME` | `rag_documents` | PGVector collection name |
| `BACKEND_HOST` | `0.0.0.0` | FastAPI bind address |
| `BACKEND_PORT` | `8000` | FastAPI port |
| `MCP_ENABLED` | `true` | Toggle MCP tools on/off |

---

### 4.7 Chainlit UI (`chainlit_app/app.py`)

A thin wrapper around the backend API. Uses `AsyncOpenAI` client pointed at `http://localhost:8000/v1` with `stream=True`.

**Conversation history:** Stored in `cl.user_session` per browser session. Entire history is sent with each request so the agent has context.

**Streaming:** Each `delta.content` token from the backend SSE stream is forwarded to the browser via `response_msg.stream_token(delta)`. Users see tokens appear as the agent reasons and responds.

**Port:** 8001. Runs independently of the backend.

---

## 5. MCP Architecture — Detailed

### 5.1 Transport Model: stdio vs HTTP

| | stdio (current) | HTTP/SSE |
|---|---|---|
| Server location | Child process (same machine) | Any networked host |
| Connection model | Parent spawns child via `command:` | Client connects to URL |
| Shared server? | No — each client spawns its own | Yes — multiple clients connect to one server |
| Process lifetime | Tied to parent process | Independent |
| Use case | Embedded / single-client | Multi-client, microservice, remote |
| Example config | `"transport": "stdio", "command": "npx"` | `"transport": "sse", "url": "http://host:3000/mcp"` |

`mcp-postgres` only supports stdio transport. To run a shared external MCP server accessible to multiple clients simultaneously, a different MCP server implementation that supports HTTP/SSE transport would be required.

### 5.2 Per-Request Session Architecture

```
Per request (current design)
  HTTP request arrives
    └── _build_agent() creates AsyncExitStack
          └── client.session("postgres") spawns: npx mcp-postgres  ← ONE process per request
                └── establishes stdio pipe
                      └── Agent makes tool calls (all share this subprocess)
                            └── subprocess executes SQL, returns results
  Request ends
    └── stack.aclose() kills subprocess cleanly
```

### 5.3 The 17 MCP Tools

Provided by `mcp-postgres`. Grouped by purpose:

**Schema Discovery**

| Tool | Description |
|---|---|
| `list_tables` | Returns all table names in the public schema |
| `describe_table` | Returns column names, types, and constraints for a table |
| `get_schema` | Returns full schema dump |
| `get_relationships` | Returns FK relationships between tables |

**Data Reading**

| Tool | Description |
|---|---|
| `query_data` | Executes a SELECT and returns results |
| `execute_raw_query` | Executes any SQL and returns results |
| `get_table_sample` | Returns a few sample rows from a table |
| `count_rows` | Returns row count for a table with optional WHERE |

**Data Writing**

| Tool | Description |
|---|---|
| `insert_data` | Inserts a row into a table |
| `update_data` | Updates rows matching a condition |
| `delete_data` | Deletes rows matching a condition |

**Metadata / Inspection**

| Tool | Description |
|---|---|
| `table_exists` | Boolean check if a table exists |
| `column_exists` | Boolean check if a column exists in a table |
| `check_certificate_cache` | Internal MCP tool |
| `get_connection_status` | Returns DB connection health |

**DDL**

| Tool | Description |
|---|---|
| `create_table` | Executes a CREATE TABLE statement |
| `alter_table` | Executes an ALTER TABLE statement |

**Who creates SQL:** The LLM (gpt-4o-mini) generates all SQL based on the schema in its system prompt and the user's question.
**Who executes SQL:** The MCP tool (`execute_raw_query` or `query_data`) sends the SQL to PostgreSQL via the `mcp-postgres` subprocess.
**Who queries documents:** The `search_documents` tool — it converts the query text to a vector embedding and runs a similarity search in PGVector. No SQL is written by the agent for this.

### 5.4 Internal vs External MCP Server

The user has a separate directory `C:\Users\yehtu\local-postgres-mcp\` with a `run-mcp.bat` that also runs `npx -y mcp-postgres`. This is **not used by the backend**.

```
C:\Users\yehtu\local-postgres-mcp\run-mcp.bat
  └── npx mcp-postgres  ← launched manually, uses stdio
        └── sits idle (no parent has connected to it via stdio)

FastAPI backend
  └── npx mcp-postgres  ← launched by lifespan(), SEPARATE process
        └── handles all agent tool calls
```

The two processes are completely independent. `run-mcp.bat` is designed for Claude Desktop (which spawns its own stdio child based on its config). To avoid confusion, you do not need to run `run-mcp.bat` when using this backend.

---

## 6. Data Flow — Query Lifecycle

### Simple database query: "How many invoices are there?"

```
1. User types in Chainlit
2. Chainlit POSTs to /v1/chat/completions (stream=true)
3. FastAPI calls rag_agent.astream_events(messages)
4. LangGraph: LLM receives system prompt (with full schema) + user message
5. LLM decides: use execute_raw_query
6. Tool call: execute_raw_query("SELECT COUNT(*) FROM invoice")
   └── JSON written to mcp-postgres stdin
   └── mcp-postgres executes SQL → returns {"count": 412}
   └── JSON read from stdout
7. LLM receives tool result, formulates response
8. "There are a total of 412 invoices in the database."
9. Tokens stream back through SSE → Chainlit → browser
```

Steps: 3 LangGraph steps. Well within recursion_limit=50.

### Complex BA query: "Top 5 artists by total revenue"

```
1–3. Same as above
4. LLM reads schema: sees invoice → invoice_line → track → album → artist chain
5. LLM decides: use execute_raw_query with a JOIN query
6. Tool call: execute_raw_query("""
       SELECT ar.artist_id, ar.name, SUM(il.unit_price * il.quantity) AS total_revenue
       FROM artist ar
       JOIN album al ON ar.artist_id = al.artist_id
       JOIN track t  ON al.album_id  = t.album_id
       JOIN invoice_line il ON t.track_id = il.track_id
       GROUP BY ar.artist_id, ar.name
       ORDER BY total_revenue DESC
       LIMIT 5
   """)
7. mcp-postgres executes, returns 5 rows
8. LLM formats as markdown table
9. Streams back through SSE
```

Steps: 3 LangGraph steps. Schema preloading means no discovery calls needed.

### Document/knowledge query: "What is RAG?"

```
1–3. Same
4. LLM decides: use search_documents
5. search_documents("What is RAG?", k=5)
   └── HuggingFace embedding model encodes query → 384-dim vector
   └── PGVector cosine similarity search in PostgreSQL
   └── Returns top-5 document chunks
6. LLM synthesises answer from document content
7. Streams back
```

---

## 7. Key Technical Decisions & Rationale

### LLM: OpenAI gpt-4o-mini (not Anthropic Claude)

**Reason:** The Anthropic account used for this prototype had insufficient credits. Switched to OpenAI `gpt-4o-mini`, which is cost-effective and competent at multi-tool ReAct loops.
**How to switch back:** Change `LLM_MODEL` in `.env` and install `langchain-anthropic`. Replace `ChatOpenAI` with `ChatAnthropic` in `rag_graph.py`.

### MCP via embedded stdio (not external HTTP server)

`mcp-postgres` only supports stdio transport. Running it as an embedded subprocess is the standard deployment model. Claude Desktop uses the same approach. An external shared MCP server would require an HTTP-transport-capable implementation.

### Schema preloading in system prompt (not via tool call)

Without preloading, the agent used `list_tables` + `describe_table` (×N tables) before every query = 4–8 extra LangGraph steps. This pushed complex queries past the recursion limit and added 2–5 seconds of latency. Preloading the ~720-token schema at startup reduces every query to the minimum tool calls needed.

### Per-request MCP session (not per-call, not server-lifetime)

Three approaches were tried before settling on the current design:

| Approach | Subprocess lifetime | Problem |
|---|---|---|
| Per-call (`get_tools()` default) | One per tool invocation | 16 subprocesses for a complex query → `GraphRecursionError` |
| Server-lifetime persistent | Entire server uptime | Subprocess dies silently; `/health` stays `ok` but all queries fail |
| **Per-request (current)** | **One HTTP request** | **Fresh connection every time; subprocess death only affects one request** |

The current design uses `AsyncExitStack` inside both the request handler and the streaming generator, with cleanup in `finally` blocks. Each request gets one guaranteed-live subprocess.

### `_StripSemicolonInterceptor` — fixing mcp-postgres SQL rejection

`mcp-postgres` rejects any SQL query containing a trailing `;` with `"Error: Multi-statement queries are not allowed"`. OpenAI models (and LLMs generally) reliably append `;` to every SQL statement they generate — this is trained behaviour that system prompt instructions cannot reliably override.

**The interceptor strips semicolons at the transport layer before SQL reaches `mcp-postgres`:**

```python
class _StripSemicolonInterceptor:
    async def __call__(self, request, handler):
        if "query" in request.args:       # attribute is .args, NOT .arguments
            request.args["query"] = (
                request.args["query"].strip().rstrip(";").strip()
            )
        return await handler(request)
```

**Critical detail:** `tool_interceptors` on `MultiServerMCPClient` only applies when using `get_tools()`. When using `client.session()` + `load_mcp_tools(session)`, interceptors must be passed explicitly to `load_mcp_tools()`:

```python
mcp_tools = await load_mcp_tools(
    session,
    tool_interceptors=[_StripSemicolonInterceptor()],  # required here
)
```

**Is this a stdio-specific problem?** No. The semicolon check is inside `mcp-postgres` application code and runs regardless of transport (stdio or HTTP). The interceptor fix is required for any transport.

**Is a separate external MCP server process the answer?** No for the semicolon issue — same code, same rejection. Yes for session stability — an HTTP-transport server would not have a dying subprocess. However, `mcp-postgres` does not support HTTP transport.

### Synchronous psycopg for schema loading (not async)

Windows uses `ProactorEventLoop` (IOCP-based). `psycopg.AsyncConnection` is incompatible with it and raises a runtime error. The synchronous call in `lifespan()` is acceptable because it runs once before the server accepts requests.

### truststore for SSL on Windows

Python's bundled SSL does not use the Windows certificate store, causing `SSLCertVerificationError` when calling `api.openai.com`. The `truststore` package injects the Windows store into Python's SSL context. Called once at the top of `main.py` before any imports that open HTTPS connections.

### OpenAI-compatible API surface

The backend exposes `/v1/chat/completions` with the same request/response shape as the OpenAI API. This lets Chainlit use the standard `AsyncOpenAI` client pointed at `http://localhost:8000/v1`. Any other client that supports a custom `base_url` can also connect (e.g., LangChain's `ChatOpenAI`, Cursor, Open WebUI).

### Streaming-first design

The agent can take 5–30 seconds for complex queries. Without streaming, the user sees a blank screen until the full answer is ready. With SSE streaming, tokens appear as they are generated. The Chainlit UI always uses `stream=True`. The non-streaming path (`stream=False`) is available for programmatic API callers but is not suitable for interactive use on long queries.

---

## 8. Known Issues & Bugs Fixed

### GraphRecursionError on complex queries

**Root cause:** Two compounding issues:
1. Agent was calling `list_tables` + multiple `describe_table` calls before every query (no schema in prompt).
2. Each MCP tool call spawned a new subprocess (per-call mode, not persistent session).

With 16+ tool calls and overhead, LangGraph's step counter hit the limit of 25.

**Fix:** Schema preloading (eliminates discovery calls) + persistent session (eliminates per-process overhead) + recursion_limit raised to 50.

### Agent hallucinating results to match user expectations

**Example:** User asked for "17 Queen albums." DB only contains 3. Agent previously fabricated 14 more to match.

**Fix:** System prompt guardrail: "ONLY report what the database actually returns. Never use your training knowledge to supplement or fill in missing data."

### SQL column/table hallucination

**Example:** Agent used `il.album_id` (does not exist) or `albums_sales` (does not exist).

**Root cause:** No schema knowledge — agent guessed column/table names.

**Fix:** Schema preloading. Agent now knows every column in every table before writing any SQL.

### `.env` not found / API key missing

**Root cause:** `config.py` resolved `.env` relative to CWD (`backend/`). Key was in project root.

**Fix:** `Path(__file__).parent.parent / ".env"` — always resolves to project root regardless of launch directory.

### MCP `aclose()` AttributeError

`MultiServerMCPClient` v0.3.0 does not have `aclose()`. The old cleanup code `await app.state.mcp_client.aclose()` raised `AttributeError` silently (caught by bare `except`), leaking the subprocess.

**Fix:** Replaced with `AsyncExitStack.aclose()` which calls `client.session().__aexit__()` correctly.

### `MultiServerMCPClient.__aenter__` NotImplementedError

The class intentionally raises `NotImplementedError` in v0.1.0+ if used as `async with MultiServerMCPClient(...)`. The error message in the exception explains the correct alternatives.

**Fix:** Use `client.session("postgres")` with `AsyncExitStack`, not the class-level context manager.

### Multi-statement SQL rejection (`mcp-postgres` false positive)

**Symptom:** Agent returns "I am unable to retrieve... due to limitations in executing database queries" for any complex query involving `DATE_TRUNC`, `EXTRACT`, or multi-line SQL.

**Root cause (3 compounding bugs):**

1. **`mcp-postgres` rejects trailing semicolons.** Any SQL ending with `;` triggers `"Error: Multi-statement queries are not allowed"` — even perfectly valid single statements. This is a false positive in `mcp-postgres`'s validation logic.

2. **LLMs always append semicolons.** GPT-4o-mini appends `;` to every SQL statement it generates. System prompt instructions ("never use semicolons") are not reliably followed — all 9 retry attempts in testing still had trailing semicolons.

3. **Interceptor attribute name was wrong.** First interceptor implementation used `request.arguments` — the actual field is `request.args`. The interceptor silently did nothing.

**Fix — `_StripSemicolonInterceptor` with correct attribute:**
```python
class _StripSemicolonInterceptor:
    async def __call__(self, request, handler):
        if "query" in request.args:            # .args not .arguments
            request.args["query"] = (
                request.args["query"].strip().rstrip(";").strip()
            )
        return await handler(request)
```
Passed to `load_mcp_tools(session, tool_interceptors=[...])` — not just to `MultiServerMCPClient`. After fix: interceptor fires once, semicolon stripped, query succeeds on first attempt. Verified stable across 3 consecutive runs.

### Silent MCP session death

**Symptom:** Query works immediately after server start, fails hours later with no error in logs. `/health` returns `ok`. Agent says "I am currently unable to retrieve..."

**Root cause:** Server-lifetime persistent session — the `npx mcp-postgres` subprocess dies (timeout, OOM, etc.) but FastAPI keeps running. The `/health` endpoint only checks FastAPI, not the MCP subprocess.

**Fix:** Per-request sessions via `AsyncExitStack` in `_build_agent()`. Each request spawns a fresh subprocess, uses it, then tears it down in `finally`. A dead subprocess only affects one request; the next request starts fresh.

### Interceptors not applied in session mode

**Root cause:** `tool_interceptors` passed to `MultiServerMCPClient(...)` are only applied when using `get_tools()`. When using `client.session()` + `load_mcp_tools(session)`, the client's interceptors are not propagated — they must be passed directly to `load_mcp_tools()`.

**Fix:** Always pass `tool_interceptors=` to both `MultiServerMCPClient` (for forward compatibility) and `load_mcp_tools()` (required for session mode).

---

## 9. Configuration Reference

### `.env` file (project root)

```env
POSTGRES_URL=postgresql://postgres:Password123@localhost:5432/postgres
OPENAI_API_KEY=sk-proj-...
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
COLLECTION_NAME=rag_documents
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
MCP_ENABLED=true
```

### Switching LLM to a different OpenAI model

```env
LLM_MODEL=gpt-4o
```

### Disabling MCP (vector search only)

```env
MCP_ENABLED=false
```

---

## 10. Package Versions

Key packages as of the last working state:

| Package | Version | Role |
|---|---|---|
| `fastapi` | 0.138.0 | HTTP API framework |
| `uvicorn` | 0.49.0 | ASGI server |
| `langgraph` | 1.2.6 | ReAct agent graph |
| `langgraph-prebuilt` | 1.1.0 | `create_react_agent` |
| `langchain-core` | 1.4.8 | Base abstractions |
| `langchain-openai` | 1.3.2 | OpenAI LLM binding |
| `langchain-mcp-adapters` | 0.3.0 | MCP → LangChain tool bridge |
| `langchain-postgres` | 0.0.17 | PGVector integration |
| `langchain-huggingface` | 1.2.2 | HuggingFace embeddings |
| `sentence-transformers` | 5.6.0 | `all-MiniLM-L6-v2` model |
| `psycopg` | 3.3.4 | PostgreSQL driver (sync, for schema_loader) |
| `psycopg-binary` | 3.3.4 | Binary distribution of psycopg |
| `pydantic` | 2.13.4 | Data validation |
| `pydantic-settings` | 2.14.2 | `.env` settings loading |
| `truststore` | 0.10.4 | Windows SSL certificate store injection |
| `chainlit` | 2.11.1 | Chat UI |
| `python-dotenv` | 1.2.2 | `.env` loading in Chainlit app |
| `torch` | 2.12.1 | Required by sentence-transformers |
| `transformers` | 5.12.1 | Required by sentence-transformers |
| `SQLAlchemy` | 2.0.51 | ORM used by langchain-postgres |

---

## 11. Running the Services

### Prerequisites

- Python 3.12 virtual environment at `.venv/`
- Node.js + npx in PATH (for `mcp-postgres`)
- PostgreSQL running locally on port 5432
- pgvector extension installed: `CREATE EXTENSION IF NOT EXISTS vector;`
- `.env` file at project root with valid `OPENAI_API_KEY`

### Start backend

```batch
start_service.cmd
```

Or manually:
```batch
cd backend
..\\.venv\Scripts\python.exe -u -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Expected startup output:
```
Connected to database: localhost:5432/postgres
MCP DB Server running on stdio
MCP PostgreSQL connected. Tools: ['get_schema', 'query_data', ...]
Database schema loaded.
Agent initialized with MCP tools: [...]
RAG agent ready.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Start Chainlit UI

```batch
start_chainlit.cmd
```

Then open: `http://localhost:8001`

### Health check

```
GET http://localhost:8000/health
→ {"status":"ok","model":"gpt-4o-mini"}
```

### Stop services

In each cmd window: `Ctrl+C`

Or via PowerShell:
```powershell
Get-NetTCPConnection -LocalPort 8000,8001 -State Listen |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

---

## 12. Database — Chinook Schema

The prototype is wired to the **Chinook** sample database — a digital music store.

```
artist ─────────────── album ──────────── track ──────── invoice_line ─── invoice ─── customer
  artist_id (PK)         album_id (PK)     track_id (PK)   invoice_line_id   invoice_id   customer_id
  name                   title             name            invoice_id (FK)   customer_id  first_name
                         artist_id (FK)    album_id (FK)   track_id (FK)     invoice_date last_name
                                           media_type_id   unit_price        billing_*    email
                                           genre_id        quantity          total        support_rep_id (FK)
                                           composer
                                           milliseconds
                                           bytes
                                           unit_price

genre                media_type          employee              playlist ──── playlist_track
  genre_id (PK)       media_type_id (PK)  employee_id (PK)      playlist_id    playlist_id (FK)
  name                name                last_name             name           track_id (FK)
                                          first_name
                                          title
                                          reports_to (FK → employee)
                                          hire_date / birth_date
                                          address / city / ...
```

**Table row counts (approximate):**

| Table | Rows |
|---|---|
| `invoice` | 412 |
| `invoice_line` | 2,240 |
| `track` | 3,503 |
| `album` | 347 |
| `artist` | 275 |
| `customer` | 59 |
| `employee` | 8 |
| `genre` | 25 |
| `media_type` | 5 |
| `playlist` | 18 |
| `playlist_track` | 8,715 |

**Note on Queen:** The Chinook DB contains only 3 Queen albums (`Queen`, `Queen For Life`, `News Of The World`). If a user asks about "17 Queen albums," the agent will report what is actually in the database (3) and will not supplement with external knowledge.

---

## 5.5 Adding More Tools

There are three ways to extend the agent's capabilities:

### Option 1: Custom MCP server (reusable across clients)

Write a Python MCP server and register it alongside the existing postgres server in `main.py`:

```python
mcp_client = MultiServerMCPClient({
    "postgres": { ... },          # existing
    "my_tools": {                 # new server
        "command": "python",
        "args": ["C:/path/to/my_mcp_server.py"],
        "transport": "stdio",
    }
})
```

The server itself:

```python
# my_mcp_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-tools")

@mcp.tool()
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email notification."""
    # implementation here
    return f"Email sent to {to}"

if __name__ == "__main__":
    mcp.run()
```

This approach makes the tools reusable by Claude Desktop or any other MCP-compatible client.

### Option 2: LangChain `@tool` (fastest, backend-only)

Add a decorated function directly in `rag_graph.py` and include it in the tools list. No MCP protocol involved:

```python
from langchain_core.tools import tool

@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """Get the current exchange rate between two currencies."""
    # call an external API, read a file, run business logic
    ...

def create_rag_agent(mcp_tools=None, db_schema=None):
    tools = [search_documents, get_exchange_rate]   # add here
    if mcp_tools:
        tools.extend(mcp_tools)
    ...
```

### Option 3: Connect to an external HTTP MCP server

If a third-party service exposes an MCP-compatible HTTP endpoint:

```python
mcp_client = MultiServerMCPClient({
    "postgres": { ... },
    "external": {
        "transport": "streamable_http",
        "url": "http://some-service:3000/mcp",
    }
})
```

### When to use each option

| Situation | Best option |
|---|---|
| Custom business logic, needed quickly | Option 2 — LangChain `@tool` |
| Tools reusable by Claude Desktop or other MCP clients | Option 1 — custom MCP server |
| Third-party service already has MCP support | Option 3 — HTTP transport |
| Adding a second database (MySQL, Redis, etc.) | Option 1 with appropriate MCP server |
| Any SQL report from the existing database | No new tool needed — just ask |

---

## 6. LLM Intelligence — How Reports Work Without Custom Code

A common misconception: *"Do I need to write a new tool for every report?"*

**No.** Tools are capabilities, not reports. The LLM generates the SQL at runtime.

### The distinction

| Concept | What it is | Written by |
|---|---|---|
| **Tool** | A generic capability — e.g., "run any SQL" | Developer (once) |
| **Report** | A specific question answered by the LLM writing SQL | Nobody — LLM does it |

The tool `execute_raw_query` can run any SQL statement. The LLM decides what SQL to write based on the user's question and the schema in its system prompt. This means every possible SQL report is available without writing any new code.

### What the LLM actually does

```
User: "Show me monthly revenue trend for 2023 by country"
            ↓
   LLM reads preloaded schema
   (knows: invoice.invoice_date, invoice.total, invoice.billing_country)
            ↓
   LLM writes SQL on the spot:

   SELECT DATE_TRUNC('month', invoice_date) AS month,
          billing_country,
          SUM(total) AS revenue
   FROM invoice
   WHERE invoice_date BETWEEN '2023-01-01' AND '2023-12-31'
   GROUP BY month, billing_country
   ORDER BY month, revenue DESC
            ↓
   execute_raw_query runs it against PostgreSQL
            ↓
   LLM formats the result as a readable table
            ↓
   Answer streams back to the user
```

No SQL was written by a developer. No new tool was added. The user just asked.

### The pipeline

```
Natural language question
        ↓
   LLM reads schema          (from system prompt — preloaded at startup)
        ↓
   LLM writes SQL            ← this is the intelligence
        ↓
   Tool executes SQL         ← this is just a database driver
        ↓
   LLM formats result
        ↓
   Answer in plain English / markdown table
```

### When you do need a new tool

Only when you need a capability that SQL cannot provide:

| Need | Requires new tool? |
|---|---|
| New SQL report or analysis | No — just ask |
| Data from a different database | Yes — new MCP server |
| Send results by email | Yes — email tool |
| Export to Excel or PDF | Yes — export tool |
| Pull data from an external REST API | Yes — API tool |
| Trigger a downstream workflow | Yes — webhook tool |
| Aggregate across two databases | Yes — or join via a view |

The rule: if the answer lives in the connected database, the LLM can get it with no new code. You add tools only when you need a new **type of action** beyond reading data.
