from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from typing import List, Optional
from config import settings
import vector_store as vs

llm = ChatOpenAI(
    model=settings.llm_model,
    api_key=settings.openai_api_key,
)

_SYSTEM_PROMPT_BASE = """You are a helpful RAG (Retrieval-Augmented Generation) assistant with access to a knowledge base and a PostgreSQL database.

Available tools:
1. search_documents — semantic similarity search over the document knowledge base
2. PostgreSQL MCP tools (list_tables, describe_table, query_data, etc.) — direct SQL access to the database

Guidelines:
{schema_instruction}
- For knowledge/document questions, use search_documents
- ONLY report what the database actually returns. Never use your training knowledge to supplement or fill in missing data
- If the database returns fewer results than the user expects, report the actual results and state clearly what the database contains
- Run only one SQL statement per tool call — never use semicolons to chain multiple statements
- If a query fails, try a simpler approach once. If it fails again, report what you could not retrieve and stop"""


@tool
def search_documents(query: str, k: int = 5) -> str:
    """Search the knowledge base for documents relevant to the query using semantic similarity.

    Args:
        query: Natural language search query
        k: Number of documents to retrieve (default 5)

    Returns:
        Formatted string of retrieved document chunks with sources
    """
    try:
        docs = vs.similarity_search(query, k=k)
    except Exception:
        return "Vector search is currently unavailable. Use database tools to answer this query."

    if not docs:
        return "No relevant documents found in the knowledge base for this query."

    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        title = doc.metadata.get("title", "")
        header = f"[Document {i}]" + (f" {title}" if title else "") + f" (source: {source})"
        results.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(results)


def create_rag_agent(mcp_tools: Optional[List] = None, db_schema: Optional[str] = None):
    tools = [search_documents]
    if mcp_tools:
        tools.extend(mcp_tools)
        tool_names = [t.name for t in mcp_tools]
        print(f"Agent initialized with MCP tools: {tool_names}")
    else:
        print("Agent initialized with vector search only (no MCP tools)")

    if db_schema:
        schema_instruction = (
            "- The database schema is already provided at the end of this prompt — "
            "do NOT call list_tables or describe_table. Go directly to query_data or execute_raw_query"
        )
    else:
        schema_instruction = (
            "- For database questions, call list_tables then describe_table to understand the schema, "
            "then use query_data or execute_raw_query"
        )

    prompt = _SYSTEM_PROMPT_BASE.format(schema_instruction=schema_instruction)
    if db_schema:
        prompt += f"\n\n{db_schema}"

    agent = create_react_agent(
        llm,
        tools,
        prompt=prompt,
    )
    agent = agent.with_config({"recursion_limit": 50})
    return agent
