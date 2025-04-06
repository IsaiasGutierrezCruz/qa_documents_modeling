from abc import ABC, abstractmethod

from llama_index.core.node_parser import MarkdownNodeParser, NodeParser, SentenceSplitter
from llama_index.core.schema import Document, TextNode
from pydantic import BaseModel


class ChunkingStrategy(ABC, BaseModel):
    """Class to define a strategy for chunking text."""

    parser: NodeParser

    @abstractmethod
    def get_chunks(self, data: list[Document]) -> list[TextNode]:
        """
        Get chunks from the text.

        Parameters
        ----------
        data : list[Document]
            The data to chunk.

        Returns
        -------
            list[TextNode]: A list of text nodes containing the chunks.
        """


class ChunkingStrategyMarkdown(ChunkingStrategy):
    """Class to define a strategy for chunking text into markdown nodes."""

    parser: MarkdownNodeParser = MarkdownNodeParser()

    def get_chunks(self, data: list[Document]) -> list[TextNode]:
        """Get chunks from the text."""
        return self.parser.get_nodes_from_documents(data)


class ChunkingStrategySentence(ChunkingStrategy):
    """Class to define a strategy for chunking text into sentences."""

    chunck_size: int = 500
    chunck_overlap: int = 20
    parser: SentenceSplitter | None = None

    def get_chunks(self, data: list[Document]) -> list[TextNode]:
        """Get chunks from the text."""
        self.parser = SentenceSplitter(
            chunk_size=self.chunck_size,
            chunk_overlap=self.chunck_overlap,
        )
        return self.parser.get_nodes_from_documents(data)


CHUNKCING_STRATEGIES = {
    "markdown": ChunkingStrategyMarkdown,
    "sentence": ChunkingStrategySentence,
}
