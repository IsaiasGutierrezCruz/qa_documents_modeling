import math
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from hamilton.function_modifiers import extract_fields
from huggingface_hub import login
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.finetuning import OpenAIFinetuneEngine
from llama_index.llms.openai import OpenAI

from qa_documents_modeling.constants import DATA_SET_PATHS
from qa_documents_modeling.data_sets.create_data_from_file import (
    DataSetGenerator,
)
from qa_documents_modeling.data_sets.format_mistral_data import (
    format_mistral_data_raft,
    format_mistral_data_rag,
)
from qa_documents_modeling.modeling_configs import ModelingConfigs
from qa_documents_modeling.nlp.chunk_strategies import CHUNKCING_STRATEGIES
from qa_documents_modeling.utils.file_management import (
    read_string_from_file,
    write_string_to_file,
)
from qa_documents_modeling.utils.utils import extract_context, extract_question
from src.qa_documents_modeling.model_creation.fine_tune_mistral import (
    FineTunneMistralModel,
    MistralModel,
)


@extract_fields(
    dict(  # noqa: C408
        train_questions=pd.DataFrame,
        test_questions=pd.DataFrame,
        golden_reference_train_rag=list[dict],
        golden_reference_test_rag=list[dict],
        raft_train_reference_noise=list[dict],
        raft_test_reference_noise=list[dict],
        raft_test_reference=list[dict],
        chunks=list[TextNode],
    ),
)
def data_set_creation(cfg: ModelingConfigs, home: Path) -> dict:
    """
    Create the data sets.

    Parameters
    ----------
    cfg: ModelingConfigs
        The modeling configs.
    home: Path
        The home directory.

    Returns
    -------
    dict
        The data sets.
    """
    llm = OpenAI(
        model=cfg.question_generation.question_generator_llm,
        temperature=cfg.question_generation.temperature,
    )
    data_set_gen = DataSetGenerator(
        root_path=home,
        strategy=cfg.question_generation.ocr,
        model=llm,
    )
    train_questions, test_questions = data_set_gen.get_questions(
        chuck_strategy=CHUNKCING_STRATEGIES[cfg.question_generation.chunking_strategy](),
        test_size=cfg.question_generation.test_size,
        number_of_questions_per_node=cfg.question_generation.number_of_questions_per_node,
    )

    (
        golden_reference_train_rag,
        golden_reference_test_rag,
        raft_train_reference_noise,
        raft_test_reference_noise,
        golden_reference_test_raft,
    ) = data_set_gen.generate_golden_reference(
        model=cfg.question_generation.question_generator_llm,
        temperature=cfg.question_generation.temperature,
        embedding_model=cfg.question_generation.embedding_model,
        similarity_top_k_embeddings=cfg.question_generation.similarity_top_k_embeddings,
        similarity_top_k_embeddings_raft=cfg.question_generation.similarity_top_k_embeddings_raft,
    )
    return {
        "train_questions": train_questions,
        "test_questions": test_questions,
        "golden_reference_train_rag": golden_reference_train_rag,
        "golden_reference_test_rag": golden_reference_test_rag,
        "raft_train_reference_noise": raft_train_reference_noise,
        "raft_test_reference_noise": raft_test_reference_noise,
        "raft_test_reference": golden_reference_test_raft,
        "chunks": data_set_gen.chunks,
    }


def format_test_reference(
    home: Path,
    golden_reference_test_rag: list[dict],
    raft_test_reference: list[dict],
    raft_test_reference_noise: list[dict],
) -> pd.DataFrame:
    if not (home / DATA_SET_PATHS.path_file_eval_reference_test).is_file():
        questions = []
        answers = []
        contexts = []
        for element in golden_reference_test_rag:
            question = extract_question(element["messages"][1]["content"], "rag")
            answer = element["messages"][2]["content"]
            context = extract_context(element["messages"][1]["content"], "rag")
            questions.append(question)
            answers.append(answer)
            contexts.append(context)
        df_rag = pd.DataFrame(
            {"question": questions, "answer_gpt4o": answers, "context": contexts},
        )
        df_rag["context_noise"] = None
        df_rag["experiment"] = "rag"

        questions = []
        answers = []
        contexts = []
        contexts_noise = []
        for element, element_noise in zip(
            raft_test_reference,
            raft_test_reference_noise,
            strict=True,
        ):
            question = extract_question(element["messages"][1]["content"], "raft")
            answer = element["messages"][2]["content"]
            context = extract_context(element["messages"][1]["content"], "raft")
            context_noise = extract_context(
                element_noise["messages"][1]["content"],
                "raft",
            )
            questions.append(question)
            answers.append(answer)
            contexts.append(context)
            contexts_noise.append(context_noise)
        df_raft = pd.DataFrame(
            {
                "question": questions,
                "answer_gpt4o": answers,
                "context": contexts,
                "context_noise": contexts_noise,
            },
        )
        df_raft["experiment"] = "raft"

        df_reference = pd.concat([df_rag, df_raft], axis=0)
        df_reference.to_csv(
            home / DATA_SET_PATHS.path_file_eval_reference_test, index=False
        )
    else:
        df_reference = pd.read_csv(home / DATA_SET_PATHS.path_file_eval_reference_test)
    return df_reference


@extract_fields(
    dict(  # noqa: C408
        gpt_without_ft=OpenAI,
        gpt_with_ft=OpenAI,
    ),
)
def fine_tune_rag_open_ai(
    cfg: ModelingConfigs,
    home: Path,
    golden_reference_train_rag: list[dict],
) -> dict:
    _ = golden_reference_train_rag
    if not (home / DATA_SET_PATHS.path_fine_tunned_model_gpt).is_file():
        finetune_engine = OpenAIFinetuneEngine(
            cfg.fine_tune_gpt.model_name,
            home / DATA_SET_PATHS.path_golden_reference_train,
        )
        finetune_engine.finetune()
        model_name = finetune_engine.get_current_job().fine_tuned_model
        write_string_to_file(model_name, home / DATA_SET_PATHS.path_fine_tunned_model_gpt)
        model_ft = finetune_engine.get_finetuned_model()
        model = OpenAI(
            model=cfg.fine_tune_gpt.model_name, temperature=cfg.fine_tune_gpt.temperature
        )
    else:
        model_name = read_string_from_file(
            home / DATA_SET_PATHS.path_fine_tunned_model_gpt
        )
        model_ft = OpenAI(model=model_name, temperature=cfg.fine_tune_gpt.temperature)
        model = OpenAI(
            model=cfg.fine_tune_gpt.model_name,
            temperature=cfg.fine_tune_gpt.temperature,
        )
    return {
        "gpt_without_ft": model,
        "gpt_with_ft": model_ft,
    }


def create_predictions_open_ai_models(
    cfg: ModelingConfigs,
    home: Path,
    chunks: list[TextNode],
    gpt_without_ft: OpenAI,
    gpt_with_ft: OpenAI,
    test_questions: pd.DataFrame,
) -> pd.DataFrame:
    if not (home / DATA_SET_PATHS.path_file_eval_gpt_35_models_rag).is_file():
        embed_model = HuggingFaceEmbedding(
            model_name=cfg.question_generation.embedding_model,
        )
        Settings.embed_model = embed_model
        index = VectorStoreIndex(chunks)

        list_dfs = []
        for model, model_name in [
            (gpt_without_ft, "gpt_35_without_ft"),
            (gpt_with_ft, "gpt_35_with_ft"),
        ]:
            query_engine = index.as_query_engine(
                similarity_top_k=cfg.question_generation.similarity_top_k_embeddings,
                llm=model,
            )
            contexts = []
            answers = []

            for question in test_questions["query"].to_list():
                response = query_engine.query(question)
                contexts.append([x.node.get_content() for x in response.source_nodes])
                answers.append(str(response))
            df_info = pd.DataFrame(
                {
                    "question": test_questions["query"].to_list(),
                    "context": contexts,
                    f"answer_{model_name}": answers,
                }
            )
            df_info["experiment"] = "rag"
            list_dfs.append(df_info)

        df_preds = list_dfs[0].merge(
            list_dfs[1][
                ["question"]
                + [
                    col_name
                    for col_name in list_dfs[1].columns
                    if col_name.startswith("answer_")
                ]
            ],
            on="question",
            how="left",
        )
        df_preds.to_csv(
            home / DATA_SET_PATHS.path_file_eval_gpt_35_models_rag,
            index=False,
        )
    else:
        df_preds = pd.read_csv(home / DATA_SET_PATHS.path_file_eval_gpt_35_models_rag)
        df_preds["context"] = df_preds["context"].apply(lambda x: eval(x))
    return df_preds


@extract_fields(
    dict(  # noqa: C408
        test_raft_noise_mistral=pd.DataFrame,
        test_raft_mistral=pd.DataFrame,
        test_ratg_mistral=pd.DataFrame,
    ),
)
def format_train_test_set_mistral_rag(
    cfg: ModelingConfigs,
    home: Path,
    golden_reference_test_rag: list[dict],
    golden_reference_train_rag: list[dict],
    raft_train_reference_noise: list[dict],
    raft_test_reference_noise: list[dict],
    raft_test_reference: list[dict],
) -> dict:
    if not (home / DATA_SET_PATHS.path_file_formatted_test_set_mistral_rag).is_file():
        login(os.getenv("HF_TOKEN"))
        train_list, _ = format_mistral_data_rag(
            data=golden_reference_train_rag, type="train"
        )
        val_size = cfg.mistral_configs.val_size
        val_size = math.floor(len(train_list) * val_size)
        val_list = train_list[:val_size]
        train_list = train_list[val_size:]
        data_with_val = DatasetDict(
            {
                "train": Dataset.from_dict({"example": train_list}),
                "test": Dataset.from_dict({"example": val_list}),
            }
        )
        data_with_val.push_to_hub(DATA_SET_PATHS.path_file_formatted_train_set_mistral_rag)

        test_list, questions = format_mistral_data_rag(
            data=golden_reference_test_rag, type="test"
        )
        test_list = pd.DataFrame({"question": questions, "query": test_list})
        test_list.to_csv(
            home / DATA_SET_PATHS.path_file_formatted_test_set_mistral_rag,
            index=False,
        )

        train_list_raft, _ = format_mistral_data_raft(
            data=raft_train_reference_noise,
            type="train",
        )
        val_size = cfg.mistral_configs.val_size
        val_size = math.floor(len(train_list_raft) * val_size)
        val_list_raft = train_list_raft[:val_size]
        train_list_raft = train_list_raft[val_size:]

        data_with_val_raft = DatasetDict(
            {
                "train": Dataset.from_dict({"example": train_list_raft}),
                "test": Dataset.from_dict({"example": val_list_raft}),
            }
        )
        data_with_val_raft.push_to_hub(
            DATA_SET_PATHS.path_file_formatted_train_set_mistral_raft,
        )

        test_list_raft, questions_raft = format_mistral_data_raft(
            data=raft_test_reference,
            type="test",
        )
        test_list_raft = pd.DataFrame(
            {"question": questions_raft, "query": test_list_raft}
        )
        test_list_raft.to_csv(
            home / DATA_SET_PATHS.path_file_formatted_test_set_mistral_raft,
            index=False,
        )

        test_list_raft_noise, questions_raft_noise = format_mistral_data_raft(
            data=raft_test_reference_noise,
            type="test",
        )
        test_list_raft_noise = pd.DataFrame(
            {"question": questions_raft_noise, "query": test_list_raft_noise}
        )
        test_list_raft_noise.to_csv(
            home / DATA_SET_PATHS.path_file_formatted_test_set_mistral_raft_noise,
            index=False,
        )
    else:
        test_list_raft_noise = pd.read_csv(
            home / DATA_SET_PATHS.path_file_formatted_test_set_mistral_raft_noise
        )
        test_list_raft = pd.read_csv(
            home / DATA_SET_PATHS.path_file_formatted_test_set_mistral_raft
        )
        test_list = pd.read_csv(
            home / DATA_SET_PATHS.path_file_formatted_test_set_mistral_rag
        )
    return {
        "test_raft_noise_mistral": test_list_raft_noise,
        "test_raft_mistral": test_list_raft,
        "test_ratg_mistral": test_list,
    }


# def fine_tune_mistral(
#     cfg: ModelingConfigs,
#     home: Path,
#     golden_reference_train_rag: list[dict],
#     raft_train_reference_noise: list[dict],
# ) -> None:
#     # load model from hub
#     if not (home / DATA_SET_PATHS.path_file_tunned_model_mistral).is_file():
#         _ = golden_reference_train_rag
#         __ = raft_train_reference_noise

#         # rag
#         mistral_model = MistralModel(
#             model_base=cfg.mistral_configs.base_model,
#             pre_trained_model=None,
#         )
#         mistral_model.load_model()
#         fine_tune_rag = FineTunneMistralModel(
#             model=mistral_model.model,
#             data_set_url=DATA_SET_PATHS.path_file_formatted_train_set_mistral_rag,
#             tokenizer=mistral_model.tokenizer,
#             final_model_name=cfg.mistral_configs.fine_tunned_model_rag,
#         )
#         fine_tune_rag.fine_tune_model(cfg.mistral_configs)

#         # raft
#         mistral_model_raft = MistralModel(
#             model_base=cfg.mistral_configs.base_model,
#             pre_trained_model=None,
#         )
#         mistral_model_raft.load_model()
#         fine_tune_raft = FineTunneMistralModel(
#             model=mistral_model_raft.model,
#             data_set_url=DATA_SET_PATHS.path_file_formatted_train_set_mistral_raft,
#             tokenizer=mistral_model_raft.tokenizer,
#             final_model_name=cfg.mistral_configs.fine_tunned_model_raft,
#         )
#         fine_tune_raft.fine_tune_model(cfg.mistral_configs)
#     mistral_rag_ft = MistralModel(
#         model_base=cfg.mistral_configs.base_model,
#         pre_trained_model=cfg.mistral_configs.fine_tunned_model_rag,
#     )
#     mistral_rag_ft.load_model()
#     mistral_raft_ft = MistralModel(
#         model_base=cfg.mistral_configs.base_model,
#         pre_trained_model=cfg.mistral_configs.fine_tunned_model_raft,
#     )
#     mistral_raft_ft.load_model()

