# RAG Prototype MCP

A full RAG pipeline with LangGraph, PostgreSQL (pgvector + MCP), FastAPI (OpenAI-compatible), and Chainlit.

## Architecture

```
Chainlit UI (port 8001)
    │  OpenAI-format HTTP
    ▼
FastAPI backend (port 8000)  ──► /v1/chat/completions
    │
    ▼
LangGraph ReAct Agent
    ├── search_documents tool  ──► pgvector (semantic search)
    └── PostgreSQL MCP tools   ──► @modelcontextprotocol/server-postgres
                                       └── query, list_tables, describe_table, ...
```

## Requirements

- Python 3.11+
- Node.js 18+ (for MCP PostgreSQL server via npx)
- PostgreSQL with pgvector extension

## Quick Start

### 1. Configure environment

Edit `.env` at the project root and fill in your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Set up the database

```bash
# From project root
python setup_db.py
```

### 4. Ingest sample documents

```bash
# From project root — loads 5 built-in sample docs
python ingest_sample.py

# Or load your own text files:
python ingest_sample.py path/to/docs/
```

### 5. Start the backend

```bash
cd backend
python main.py
# Starts on http://localhost:8000
# API docs: http://localhost:8000/docs
```

### 6. Install and start the Chainlit app

```bash
cd chainlit_app
pip install -r requirements.txt
chainlit run app.py --port 8001
# Opens http://localhost:8001
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| GET | /v1/models | List available models |
| POST | /v1/chat/completions | Chat (OpenAI format, supports streaming) |

### Example request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rag-model",
    "messages": [{"role": "user", "content": "What is LangGraph?"}],
    "stream": false
  }'
```

## Connecting to OpenWebUI

Add a custom OpenAI API connection in OpenWebUI:
- **API Base URL**: `http://localhost:8000/v1`
- **API Key**: `not-required`
- **Model**: `rag-model`

## Adding Your Own Documents

```python
# ingest_sample.py supports text files
python ingest_sample.py my_docs/

# Or programmatically:
from backend.vector_store import add_documents
from langchain_core.documents import Document

docs = [Document(page_content="...", metadata={"source": "my-source"})]
add_documents(docs)
```

## MCP PostgreSQL Tools

When Node.js is available, the agent gains direct SQL access:

- `query` — Execute SQL SELECT queries
- `list_tables` — List all tables in the database
- `describe_table` — Get table schema
- `list_schemas` — List all schemas

If Node.js / npx is unavailable, the system falls back to vector search only.
