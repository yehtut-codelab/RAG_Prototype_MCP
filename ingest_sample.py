"""Ingest documents into the vector store.

Usage:
    python ingest_sample.py                  # load built-in samples
    python ingest_sample.py path/to/file.txt # load a text file
    python ingest_sample.py path/to/dir/     # load all .txt files in directory
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from vector_store import add_documents
from config import settings  # noqa: E402

SAMPLE_DOCS = [
    Document(
        page_content=(
            "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
            "It extends LangChain with the ability to build cyclic graphs, which allows for more "
            "complex agent behaviors including loops, conditionals, and multi-step reasoning. "
            "Key concepts: StateGraph, nodes, edges, conditional edges, and checkpointing."
        ),
        metadata={"source": "sample", "title": "LangGraph Overview"},
    ),
    Document(
        page_content=(
            "Model Context Protocol (MCP) is an open standard that enables AI assistants to "
            "securely connect to external data sources and tools. MCP servers expose resources, "
            "tools, and prompts that AI clients can consume. The PostgreSQL MCP server "
            "(@modelcontextprotocol/server-postgres) provides tools: query, list_tables, "
            "describe_table, and list_schemas."
        ),
        metadata={"source": "sample", "title": "MCP Overview"},
    ),
    Document(
        page_content=(
            "pgvector is a PostgreSQL extension for vector similarity search. It supports "
            "exact and approximate nearest-neighbor search. Key operators: <-> (L2 distance), "
            "<#> (negative inner product), <=> (cosine distance). Create an index with: "
            "CREATE INDEX ON items USING ivfflat (embedding vector_cosine_ops)."
        ),
        metadata={"source": "sample", "title": "pgvector Reference"},
    ),
    Document(
        page_content=(
            "RAG (Retrieval-Augmented Generation) combines a retrieval system with a generative "
            "LLM. The pipeline: (1) embed the user query, (2) retrieve top-k similar chunks "
            "from a vector store, (3) inject retrieved context into the LLM prompt, (4) generate "
            "a grounded answer. This reduces hallucinations and allows the model to use "
            "up-to-date or domain-specific knowledge."
        ),
        metadata={"source": "sample", "title": "RAG Architecture"},
    ),
    Document(
        page_content=(
            "FastAPI is a modern, fast Python web framework for building APIs with automatic "
            "OpenAPI documentation. It uses Pydantic for data validation and supports async "
            "operations natively. For OpenAI-compatible APIs, implement POST /v1/chat/completions "
            "and GET /v1/models endpoints following the OpenAI Chat Completions specification."
        ),
        metadata={"source": "sample", "title": "FastAPI for AI APIs"},
    ),
]


def load_text_file(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    chunks = [text[i : i + 1000] for i in range(0, len(text), 800)]
    return [
        Document(
            page_content=chunk,
            metadata={"source": str(path), "title": path.stem},
        )
        for chunk in chunks
        if chunk.strip()
    ]


def main():
    docs: list[Document] = []

    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
        if target.is_dir():
            for f in sorted(target.glob("**/*.txt")):
                docs.extend(load_text_file(f))
                print(f"Loaded {f}")
        elif target.is_file():
            docs.extend(load_text_file(target))
            print(f"Loaded {target}")
        else:
            print(f"Path not found: {target}")
            sys.exit(1)
    else:
        docs = SAMPLE_DOCS
        print(f"Loading {len(docs)} built-in sample documents...")

    if not docs:
        print("No documents to ingest.")
        return

    print(f"Ingesting {len(docs)} document(s) into vector store (this embeds and stores them)...")
    ids = add_documents(docs)
    print(f"Done. {len(ids)} chunks stored in collection '{settings.collection_name}'.")


if __name__ == "__main__":
    main()
