from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List
from config import settings

_embeddings: HuggingFaceEmbeddings | None = None
_vector_store: PGVector | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        print(f"Loading embedding model: {settings.embedding_model}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vector_store() -> PGVector:
    global _vector_store
    if _vector_store is None:
        _vector_store = PGVector(
            embeddings=get_embeddings(),
            collection_name=settings.collection_name,
            connection=settings.postgres_sqlalchemy_url,
            use_jsonb=True,
        )
    return _vector_store


def add_documents(docs: List[Document]) -> List[str]:
    return get_vector_store().add_documents(docs)


def similarity_search(query: str, k: int = 5) -> List[Document]:
    return get_vector_store().similarity_search(query, k=k)


def similarity_search_with_score(query: str, k: int = 5):
    return get_vector_store().similarity_search_with_score(query, k=k)
