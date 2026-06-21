# RAG + MCP Query Flow

## Who Does What When You Ask a Question

```
User Question
     │
     ▼
┌─────────────┐
│  GPT-4o-mini │  ← Creates the SQL (decides WHAT to query and writes the SQL)
│  (LangGraph) │  ← Also decides WHICH tool to call
└──────┬──────┘
       │  calls tool with SQL string
       ▼
┌─────────────┐
│  MCP Client  │  ← Just a messenger, passes the SQL to the MCP server
│ (LangChain)  │
└──────┬──────┘
       │  sends over stdio
       ▼
┌─────────────┐
│  mcp-postgres│  ← Runs the SQL against PostgreSQL (executes it)
│  (Node.js)   │
└──────┬──────┘
       │  raw results (rows/data)
       ▼
┌─────────────┐
│  GPT-4o-mini │  ← Reads the results and writes the answer in plain English
└──────┬──────┘
       │
       ▼
   User Answer
```

## Roles Summary

| Component | Role |
|---|---|
| **GPT-4o-mini** | Creates the SQL, reads results, writes the human-readable answer |
| **mcp-postgres** | Executes the SQL against PostgreSQL — no intelligence, just runs it |
| **LangGraph** | Orchestrates the loop (call model → call tool → call model again) |
| **MCP Client** | Transport layer between Python and the Node.js MCP server |
| **`search_documents`** | Queries text only — converts question to embeddings and finds similar docs (no SQL) |

> The LLM never touches the database directly. It only ever sees **text in, text out**.

---

## MCP Tools (17 total)

Provided by `mcp-postgres` running locally at `C:\Users\yehtu\local-postgres-mcp`.

### Schema Discovery
| Tool | Purpose |
|---|---|
| `list_tables` | List all tables in the database |
| `describe_table` | Get columns, types, and constraints for a table |
| `get_schema` | Full schema overview of all tables |
| `get_table_sample` | Preview a few rows from a table |
| `get_relationships` | Show foreign key relationships between tables |

### Read
| Tool | Purpose |
|---|---|
| `query_data` | Run SELECT queries |
| `execute_raw_query` | Run any arbitrary SQL |
| `count_rows` | Count rows in a table (with optional filter) |
| `table_exists` | Check if a table exists |
| `column_exists` | Check if a column exists in a table |

### Write
| Tool | Purpose |
|---|---|
| `insert_data` | Insert rows into a table |
| `update_data` | Update existing rows |
| `delete_data` | Delete rows |

### DDL
| Tool | Purpose |
|---|---|
| `create_table` | Create a new table |
| `alter_table` | Modify a table's structure |

### Diagnostics
| Tool | Purpose |
|---|---|
| `get_connection_status` | Check if the DB connection is alive |
| `check_certificate_cache` | Check SSL certificate cache status |

---

## Vector Search vs MCP

| | Vector Search (`search_documents`) | MCP Tools |
|---|---|---|
| **What it searches** | Document knowledge base (ingested PDFs, text files) | PostgreSQL database directly |
| **How it works** | Converts query to embeddings, finds semantically similar chunks | Generates and executes SQL |
| **When to use** | "What does the manual say about X?" | "How many invoices does customer Y have?" |
| **Required for DB queries?** | No | Yes |

---

## Stack

- **LLM:** `gpt-4o-mini` via `langchain-openai`
- **Agent framework:** LangGraph `create_react_agent`
- **MCP transport:** stdio (mcp-postgres spawned as child process)
- **MCP server:** `mcp-postgres` (Node.js) at `C:\Users\yehtu\local-postgres-mcp`
- **Vector store:** PGVector (`langchain-postgres`) with HuggingFace embeddings
- **API:** FastAPI on `http://0.0.0.0:8000` (OpenAI-compatible endpoints)
