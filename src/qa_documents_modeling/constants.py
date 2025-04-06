from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ConstantsDataSetPaths:
    """Constants for the data set paths."""

    # inital file'
    path_file_pdf_data_set = "data/CONTRATO_AP000000718.pdf"
    extracted_text = "data/extracted_text.pkl"
    # data sets files
    path_file_train_questions = "data/modeling/train_questions.csv"
    path_file_test_questions = "data/modeling/test_questions.csv"
    # rag data sets
    path_golden_reference_train = "data/modeling/golden_reference_train.jsonl"
    path_golden_reference_test = "data/modeling/golden_reference_test.jsonl"
    # raft data sets
    path_file_raft_train_questions = "data/modeling/raft_train_reference.jsonl"
    path_file_raft_test_questions = "data/modeling/raft_test_reference.jsonl"
    path_file_raft_train_questions_noise = (
        "data/modeling/raft_train_reference_noise.jsonl"
    )
    path_file_raft_test_questions_noise = "data/modeling/raft_test_reference_noise.jsonl"
    # fine tunned model
    path_fine_tunned_model_gpt = "data/modeling/models/fine_tunned_model_gpt.txt"
    path_file_tunned_model_mistral = "data/modeling/models/fine_tunned_model_mistral.txt"
    path_file_tunned_model_mistral_raft = (
        "data/modeling/models/fine_tunned_model_mistral_raft.txt"
    )
    # formatted data sets mistral
    path_file_formatted_train_set_mistral_rag = (
        "isaiasgutierrezcruz/qa_documents_val"
    )
    path_file_formatted_test_set_mistral_rag = (
        "data/modeling/sets/formatted_test_set_mistral_rag.csv"
    )
    path_file_formatted_train_set_mistral_raft = (
        "isaiasgutierrezcruz/qa_documents_val_raft"
    )
    path_file_formatted_test_set_mistral_raft = (
        "data/modeling/sets/formatted_test_set_mistral_raft.csv"
    )
    path_file_formatted_test_set_mistral_raft_noise = (
        "data/modeling/sets/formatted_test_set_mistral_raft_noise.csv"
    )
    # evaluation data sets
    path_file_eval_reference_test = "data/modeling/eval/eval_reference_test.csv"
    path_file_eval_gpt_35_models_rag = "data/modeling/eval/eval_gpt_35_models_rag.csv"


DATA_SET_PATHS = ConstantsDataSetPaths()


PIPELINE_STEPS = [
    "data_set_creation",
    "format_test_reference",
    "fine_tune_rag_open_ai",
    "create_predictions_open_ai_models",
    "format_train_test_set_mistral_rag",
    # "fine_tune_mistral",
]
