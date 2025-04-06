import json
import math
import os
import random
from pathlib import Path
from typing import Literal

import pandas as pd
from llama_index.core import (
    Settings,
    VectorStoreIndex,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.finetuning.callbacks import OpenAIFineTuningHandler
from llama_index.llms.openai import OpenAI
from llama_index.readers.docling import DoclingReader

from qa_documents_modeling.constants import DATA_SET_PATHS
from qa_documents_modeling.nlp.chunk_strategies import ChunkingStrategy
from qa_documents_modeling.utils.file_management import load_data, save_data


def insert_before_delimiter(
    original_string: str,
    strings_to_insert: list[str],
    delimiter: str = "\n---------------------\n",
) -> str:
    r"""
    Insert strings before a delimiter in a text string.

    Parameters
    ----------
    original_string : str
        The original text string to modify
    strings_to_insert : list[str]
        List of strings to insert before the delimiter
    delimiter : str, optional
        The delimiter string to insert before, by default "\n---------------------\n"

    Returns
    -------
    str
        Modified string with new text inserted before the delimiter

    Notes
    -----
    If the delimiter appears multiple times, the function will insert before the last occurrence.
    If the delimiter is not found, appends the first string to insert and the delimiter.
    """
    if delimiter not in original_string:
        return original_string + delimiter + strings_to_insert[0]

    # Split the string into parts using the delimiter
    parts = original_string.split(delimiter)

    # Get the text after the first delimiter
    text_after_first_delimiter = parts[1].strip() if len(parts) > 1 else ""

    # Find the appropriate string to insert
    string_to_insert = strings_to_insert[0]  # Default to first string
    for i, alt_string in enumerate(strings_to_insert):
        if text_after_first_delimiter == alt_string:
            # Use the next string in the list, or wrap around to the first if at the end
            next_index = (i + 1) % len(strings_to_insert)
            string_to_insert = strings_to_insert[next_index]
            break

    # If there's only one occurrence, handle it like before
    if len(parts) == 2:
        return parts[0] + string_to_insert + delimiter + parts[1]

    # For multiple occurrences, join all parts except the last one with the delimiter,
    return delimiter.join(parts[:-1]) + string_to_insert + delimiter + parts[-1]


def append_jsonl_safely(data, filename: str):
    """
    Append JSONL data safely with proper character encoding and escaping
    """
    try:
        # Convert to JSON string with proper escaping
        json_str = json.dumps(data, ensure_ascii=True)
        # Check if file exists and is not empty
        file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
        # Open file in append mode
        with open(filename, "a", encoding="utf-8") as f:
            # Add newline before appending if file exists and is not empty
            if file_exists:
                f.write("\n")
            f.write(json_str)
        # Verify the last line can be read back
        with open(filename, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if lines:
                json.loads(lines[-1].strip())
    except Exception as e:
        print(f"Error appending to JSONL file: {str(e)}")


def read_jsonl_file(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_questions_from_nodes(
    nodes: list[TextNode],
    model: OpenAI,
    query: str,
    path_to_save: Path,
    num_questions_per_chunk: int = 50,
) -> pd.DataFrame:
    """Create questions from nodes."""
    if path_to_save.is_file():
        return pd.read_csv(path_to_save)

    generator = RagDatasetGenerator(
        nodes=nodes,
        llm=model,
        question_gen_query=query,
        num_questions_per_chunk=num_questions_per_chunk,
    )
    dataset = generator.generate_dataset_from_nodes()

    df_qa = dataset.to_pandas()
    df_qa.to_csv(path_to_save, index=False)

    return df_qa


def create_golden_reference_from_openai(
    model: Literal["gpt-4o"],
    temperature: float,
    df_questions: pd.DataFrame,
    index: VectorStoreIndex,
    path_to_save: Path,
    similarity_top_k: int = 2,
) -> list[dict]:
    """Create a golden reference from OpenAI."""
    if not path_to_save.is_file():
        print(f"Creating golden reference for {path_to_save}")
        finetuning_handler = OpenAIFineTuningHandler()
        callback_manager = CallbackManager([finetuning_handler])
        llm = OpenAI(
            model=model,
            temperature=temperature,
        )
        (Settings.callback_manager,) = (callback_manager,)

        query_engine = index.as_query_engine(similarity_top_k=similarity_top_k, llm=llm)

        for question in df_questions["query"]:
            _ = query_engine.query(question)

        finetuning_handler.save_finetuning_events(str(path_to_save))

    return read_jsonl_file(path_to_save)


def create_golden_reference_from_openai_raft(
    model: Literal["gpt-4o"],
    temperature: float,
    df_questions: pd.DataFrame,
    index: VectorStoreIndex,
    path_to_save: Path,
    similarity_top_k: int = 1,
    chunks: list[TextNode] = None,
    path_to_save_noise: Path = None,
) -> tuple[list[dict], list[dict]]:
    """Create a golden reference from OpenAI."""
    if not path_to_save.is_file():
        print(f"Creating golden reference for {path_to_save}")
        chat_text_qa_prompt_str = """
        {% chat role="system" %}
        Eres un sistema experto de preguntas y respuestas en el que se confía en todo el mundo.\n
        Siempre responde a la pregunta utilizando la información del contexto proporcionado,
        y no conocimiento previo.\n
        Algunas reglas a seguir:\n
        1. Nunca hagas referencia directa al contexto proporcionado en tu respuesta.
        2. Evita afirmaciones como 'Basándote en el contexto, ...' o
        'La información del contexto ...' o cualquier cosa similar.
        3. La respuesta debe estar escrita en Español
        4. Primero proporciona un razonamiento para encontrar la respuesta de manera concisa y sin dar información no necesaria.
        5. En el razonamiento, si necesitas copiar y pegar algunas 1  frases del contexto, inclúyelas entre ##begin_quote## y ##end_quote##. Esto significaría que lo que esté fuera de ##begin_quote## y ##end_quote## no se copia y pega directamente del contexto.
        6. Termina tu respuesta con la respuesta final en la forma <ANSWER>: $respuesta, la respuesta debe ser concisa.
        {% endchat %}

        {% chat role="user" %}
        La información del contexto está a continuación.\n
        <context>
        {{ context_str }}\n
        <context>
        Dada la información del contexto y sin conocimiento previo,
        responde la pregunta.\n
        Pregunta: {{query_str}}\n
        Respuesta:
        {% endchat %}
        """
        text_qa_template = RichPromptTemplate(chat_text_qa_prompt_str)

        finetuning_handler = OpenAIFineTuningHandler()
        callback_manager = CallbackManager([finetuning_handler])
        llm = OpenAI(
            model=model,
            temperature=temperature,
        )
        (Settings.callback_manager,) = (callback_manager,)

        query_engine = index.as_query_engine(similarity_top_k=similarity_top_k, llm=llm)
        query_engine.update_prompts(
            {
                "response_synthesizer:text_qa_template": text_qa_template,
            }
        )

        for question in df_questions["query"]:
            _ = query_engine.query(question)

        finetuning_handler.save_finetuning_events(str(path_to_save))

        data = read_jsonl_file(path_to_save)
        chunks_copy = chunks.copy()

        for seed, element in enumerate(data):
            random.seed(seed)
            random.shuffle(chunks_copy)
            noise_text = random.sample(chunks_copy, 3)
            noise_text = [node.text for node in noise_text]
            element["messages"][1]["content"] = insert_before_delimiter(
                element["messages"][1]["content"],
                noise_text,
                delimiter="<context>",
            )
            append_jsonl_safely(element, path_to_save_noise)

    return read_jsonl_file(path_to_save_noise), read_jsonl_file(path_to_save)


class DataSetGenerator:
    """
    Class to extract text from a file using a specific strategy of OCR.

    Parameters
    ----------
    file_source : Path
        The path to the file to extract text from.
    strategy : Literal["docling"]
        The strategy to use for extracting text.
    """

    query = (
        "Eres un Asistente de IA generando preguntas para un conjunto de datos de Pregunta-Respuesta (Question-Answering). "
        "Tu tarea es formular preguntas basadas *únicamente* en el contexto de texto proporcionado.\n"
        "Cada pregunta debe:\n"
        "1. Indagar sobre un único hecho importante, concepto, definición o pieza clave de información explícitamente mencionado en el texto.\n"
        "2. Ser clara, concisa y respondible usando *únicamente* el contexto de texto proporcionado.\n"
        "3. Ser diversa, cubriendo diferentes aspectos o detalles mencionados en el texto si es posible.\n"
        "Genera la lista de preguntas, cada una en una nueva línea."
    )

    def __init__(
        self,
        root_path: Path,
        strategy: Literal["docling"],
        model: OpenAI,
    ) -> None:
        self.root_path = root_path
        self.strategy = strategy
        self.model = model

        self.chunks: list[TextNode] = []
        self.data: list[Document] = []
        self.train_set: pd.DataFrame
        self.test_set: pd.DataFrame
        self.train_questions: pd.DataFrame
        self.test_questions: pd.DataFrame

    def _extract_text(self) -> list[Document]:
        """
        Extract text from a file using a specific strategy of OCR.

        Only supported strategy "docling" for now.

        Parameters
        ----------
        file_source : str
            The path to the file to extract text from.
        strategy : Literal["docling"]
            The strategy to use for extracting text.

        Returns
        -------
            list[Document]: A list of documents containing the extracted text.
        """
        if (self.root_path / DATA_SET_PATHS.extracted_text).is_file():
            self.data = load_data(self.root_path / DATA_SET_PATHS.extracted_text)
        else:
            pdf_path = self.root_path / DATA_SET_PATHS.path_file_pdf_data_set
            if pdf_path.is_file():
                if self.strategy == "docling":
                    reader = DoclingReader()
                    data = reader.load_data(pdf_path)
                else:
                    msg = f"Strategy {self.strategy} not supported"
                    raise ValueError(msg)
            else:
                msg = f"File {pdf_path} does not exist"
                raise FileNotFoundError(msg)
            save_data(data, self.root_path / DATA_SET_PATHS.extracted_text)

            self.data = data

    def _get_chucks(self, chuck_strategy: ChunkingStrategy) -> None:
        """Get chunks from the text."""
        if self.data is None:
            msg = "Data is not loaded"
            raise ValueError(msg)
        self.chunks = chuck_strategy.get_chunks(self.data)

    def get_questions(
        self,
        chuck_strategy: ChunkingStrategy,
        test_size: int,
        number_of_questions_per_node: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get the data sets.

        Parameters
        ----------
        chuck_strategy : ChunkingStrategy
            The strategy to use for chunking the text.
        test_size : int
            The size of the test set.
        number_of_questions_per_node : int
            The number of questions to generate per node.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]: A tuple of data frames containing the train and
        test questions.
        """
        self._extract_text()
        self._get_chucks(chuck_strategy)
        random.seed(42)
        random.shuffle(self.chunks)
        test_size = math.floor(len(self.chunks) * test_size)

        test_chunks = self.chunks[:test_size]
        train_chunks = self.chunks[test_size:]

        self.train_questions = create_questions_from_nodes(
            train_chunks,
            self.model,
            self.query,
            self.root_path / DATA_SET_PATHS.path_file_train_questions,
            number_of_questions_per_node,
        )

        self.test_questions = create_questions_from_nodes(
            test_chunks,
            self.model,
            self.query,
            self.root_path / DATA_SET_PATHS.path_file_test_questions,
            number_of_questions_per_node,
        )

        return self.train_questions, self.test_questions

    def generate_golden_reference(
        self,
        model: Literal["gpt-4o"],
        temperature: float,
        embedding_model: Literal["paraphrase-multilingual-mpnet-base-v2"],
        similarity_top_k_embeddings: int,
        similarity_top_k_embeddings_raft: int,
    ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
        embed_model = HuggingFaceEmbedding(model_name=embedding_model)
        Settings.embed_model = embed_model
        index = VectorStoreIndex(self.chunks)

        # rag references
        golden_reference_train_rag = create_golden_reference_from_openai(
            model=model,
            temperature=temperature,
            df_questions=self.train_questions,
            index=index,
            path_to_save=self.root_path / DATA_SET_PATHS.path_golden_reference_train,
            similarity_top_k=similarity_top_k_embeddings,
        )

        golden_reference_test_rag = create_golden_reference_from_openai(
            model=model,
            temperature=temperature,
            df_questions=self.test_questions,
            index=index,
            path_to_save=self.root_path / DATA_SET_PATHS.path_golden_reference_test,
            similarity_top_k=similarity_top_k_embeddings,
        )

        # raft references
        golden_reference_train_raft_noise, _ = create_golden_reference_from_openai_raft(
            model=model,
            temperature=temperature,
            df_questions=self.train_questions,
            index=index,
            path_to_save=self.root_path / DATA_SET_PATHS.path_file_raft_train_questions,
            similarity_top_k=similarity_top_k_embeddings_raft,
            chunks=self.chunks,
            path_to_save_noise=self.root_path
            / DATA_SET_PATHS.path_file_raft_train_questions_noise,
        )
        golden_reference_test_raft_noise, golden_reference_test_raft = (
            create_golden_reference_from_openai_raft(
                model=model,
                temperature=temperature,
                df_questions=self.test_questions,
                index=index,
                path_to_save=self.root_path
                / DATA_SET_PATHS.path_file_raft_test_questions,
                similarity_top_k=similarity_top_k_embeddings_raft,
                chunks=self.chunks,
                path_to_save_noise=self.root_path
                / DATA_SET_PATHS.path_file_raft_test_questions_noise,
            )
        )
        return (
            golden_reference_train_rag,
            golden_reference_test_rag,
            golden_reference_train_raft_noise,
            golden_reference_test_raft_noise,
            golden_reference_test_raft,
        )

    # def get_data_sets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
    #     """Get the data sets."""

    #     # Read training data
    #     train_file = os.path.join(modeling_dir, 'golden_reference_train.jsonl')
    #     train_data = read_jsonl_file(train_file)
    #     print(f"Loaded {len(train_data)} training examples")

    #     # Read test data
    #     test_file = os.path.join(modeling_dir, 'golden_reference_test.jsonl')
    #     test_data = read_jsonl_file(test_file)
    #     print(f"Loaded {len(test_data)} test examples")
