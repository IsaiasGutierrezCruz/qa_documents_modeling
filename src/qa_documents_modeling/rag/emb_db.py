from abc import ABC, abstractmethod
from typing import Literal

from llama_index.core.schema import Document

from qa_documents_modeling.rag.llm import LLMService


class RAGEmbeddingDB(ABC):
    @abstractmethod
    def query(self, text: str) -> list[float]:
        pass


class OpenAIEmbeddingLlamaIndexDB(RAGEmbeddingDB):
    def __init__(
        self, model: LLMService, similarity_top_k: int, documents: list[Document]
    ) -> None:
        self.model = model
        self.similarity_top_k = similarity_top_k
        self.documents = documents

    def query(self, text: str) -> list[float]:
        return self.model.as_query_engine(text, self.similarity_top_k)


def get_embedding_db(
    model: LLMService,
    similarity_top_k: int,
    documents: list[Document],
    provider: Literal["openai"],
) -> RAGEmbeddingDB:
    if provider == "openai":
        return OpenAIEmbeddingLlamaIndexDB(
            model=model, similarity_top_k=similarity_top_k, documents=documents,
        )
    else:
        msg = f"Unsupported embedding provider: {provider}"
        raise ValueError(msg)
