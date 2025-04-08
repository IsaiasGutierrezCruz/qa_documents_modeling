import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Metric column name constants
CONTEXT_COUNT_PREFIX = "count_context_"
BLEU_SCORE_MEAN_PREFIX = "bleu_score_mean_"
VECTOR_SIMILARITY_MEAN_PREFIX = "vector_similarity_mean_"


def calculate_bleu_score(reference: str, candidate: str) -> float:
    return sentence_bleu(reference, candidate)


def calculate_metrics(
    df: pd.DataFrame, columns_to_evaluate: list[str], col_reference: str
) -> pd.DataFrame:
    data = df.copy()

    # Rule based evaluation
    for column in columns_to_evaluate:
        data[f"{CONTEXT_COUNT_PREFIX}{column}"] = data.apply(
            lambda row: "context" in row[column], axis=1
        )

    # Metric based evaluation
    for column in columns_to_evaluate:
        data[f"{BLEU_SCORE_MEAN_PREFIX}{column}"] = data.apply(
            lambda row: calculate_bleu_score(
                [row[col_reference].lower().split()], row[column].lower().split()
            ),
            axis=1,
        )

    # Metrics based on vector similarity
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    for column in columns_to_evaluate:
        data[f"{VECTOR_SIMILARITY_MEAN_PREFIX}{column}"] = data.apply(
            lambda row: cosine_similarity(
                model.encode(row[col_reference], show_progress_bar=False).reshape(1, -1),
                model.encode(row[column], show_progress_bar=False).reshape(1, -1),
            )[0][0],
            axis=1,
        )

    return data


def create_metrics_pivot_table(
    df: pd.DataFrame, columns_to_evaluate: list[str]
) -> pd.DataFrame:
    """
    Create a pivot table summarizing the evaluation metrics for each model.

    Args:
        df (pd.DataFrame): DataFrame containing the evaluation metrics
        columns_to_evaluate (list[str]): List of columns containing model outputs to evaluate

    Returns:
        pd.DataFrame: Pivot table with aggregated metrics for each model
    """
    # Create a list of all metric columns
    metric_columns = []
    for column in columns_to_evaluate:
        metric_columns.extend(
            [
                f"{CONTEXT_COUNT_PREFIX}{column}",
                f"{BLEU_SCORE_MEAN_PREFIX}{column}",
                f"{VECTOR_SIMILARITY_MEAN_PREFIX}{column}",
            ]
        )

    # Calculate mean values for each metric
    pivot_data = []
    for column in columns_to_evaluate:
        pivot_data.append(
            {
                "model": column,
                "context_count_mean": df[f"{CONTEXT_COUNT_PREFIX}{column}"].sum(),
                "bleu_score_mean": df[f"{BLEU_SCORE_MEAN_PREFIX}{column}"].mean(),
                "vector_similarity_mean": df[
                    f"{VECTOR_SIMILARITY_MEAN_PREFIX}{column}"
                ].mean(),
                "bleu_score_median": df[f"{BLEU_SCORE_MEAN_PREFIX}{column}"].median(),
                "vector_similarity_median": df[
                    f"{VECTOR_SIMILARITY_MEAN_PREFIX}{column}"
                ].median(),
            }
        )

    return pd.DataFrame(pivot_data)
