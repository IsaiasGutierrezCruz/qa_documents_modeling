from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class QuestionGeneration:
    """Constants for the question generator LLM."""

    question_generator_llm: Literal["gpt-4o", "gpt-3.5-turbo"]
    temperature: float
    ocr: Literal["docling"]
    test_size: float
    number_of_questions_per_node: int
    chunking_strategy: Literal["markdown", "sentence"]
    embedding_model: Literal["paraphrase-multilingual-mpnet-base-v2"]
    similarity_top_k_embeddings: int
    similarity_top_k_embeddings_raft: int


@dataclass(frozen=True)
class FineTunnedGPT:
    """Constants for the fine tuning."""

    model_name: Literal["gpt-3.5-turbo", "gpt-4o"]
    temperature: float

@dataclass(frozen=True)
class MistralConfigs:
    """Constants for the mistral configs."""

    base_model: Literal["TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"]
    val_size: float
    fine_tunned_model_rag: str
    fine_tunned_model_raft: str
    r: int
    lora_alpha: int
    lora_dropout: float
    task_type: Literal["CAUSAL_LM"]
    lr: float
    batch_size: int
    num_epochs: int

@dataclass(frozen=True)
class ModelingConfigs:
    """Constants for the modeling configs."""

    question_generation: QuestionGeneration
    fine_tune_gpt: FineTunnedGPT
    mistral_configs: MistralConfigs


MODELING_CONFIGS = ModelingConfigs(
    question_generation=QuestionGeneration(
        question_generator_llm="gpt-4o",
        temperature=0.3,
        ocr="docling",
        test_size=0.2,
        number_of_questions_per_node=3,
        chunking_strategy="sentence",
        embedding_model="paraphrase-multilingual-mpnet-base-v2",
        similarity_top_k_embeddings=2,
        similarity_top_k_embeddings_raft=1,
    ),
    fine_tune_gpt=FineTunnedGPT(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
    ),
    mistral_configs=MistralConfigs(
        base_model="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
        val_size=0.2,
        fine_tunned_model_rag="isaiasgutierrezcruz/qa_documents_ft_val",
        fine_tunned_model_raft="isaiasgutierrezcruz/qa_documents_ft_val_raft",
        # hyperparameters
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        lr=2e-4,
        batch_size=4,
        num_epochs=20,
    ),
)
