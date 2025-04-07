import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_bleu_score(reference: str, candidate: str) -> float:
    return sentence_bleu(reference, candidate)


def calculate_metrics(
    df: pd.DataFrame, columns_to_evaluate: list[str], col_reference: str
) -> pd.DataFrame:
    data = df.copy()

    # Rule based evaluation
    count_context_in_answer = []
    for column in columns_to_evaluate:
        count_context_in_answer.append(
            int(data.apply(lambda row: "context" in row[column], axis=1).sum())
        )

    # Metric based evaluation
    bleu_scores_mean = []
    bleu_scores_median = []
    for column in columns_to_evaluate:
        bleu_scores_mean.append(
            data.apply(
                lambda row: calculate_bleu_score(
                    [row[col_reference].lower().split()], row[column].lower().split()
                ),
                axis=1,
            ).mean()
        )
        bleu_scores_median.append(
            data.apply(
                lambda row: calculate_bleu_score(
                    [row[col_reference].lower().split()], row[column].lower().split()
                ),
                axis=1,
            ).median()
        )
    # Metrics based on vector similarity
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    vector_similarity_mean = []
    vector_similarity_median = []
    for column in columns_to_evaluate:
        vector_similarity_mean.append(
            data.apply(
                lambda row: cosine_similarity(
                    model.encode(row[col_reference], show_progress_bar=False).reshape(1, -1),
                    model.encode(row[column], show_progress_bar=False).reshape(1, -1),
                )[0][0],
                axis=1,
            ).mean()
        )
        vector_similarity_median.append(
            data.apply(
                lambda row: cosine_similarity(
                    model.encode(row[col_reference], show_progress_bar=False).reshape(1, -1),
                    model.encode(row[column], show_progress_bar=False).reshape(1, -1),
                )[0][0],
                axis=1,
            ).median()
        )

    return pd.DataFrame(
        {
            "model": columns_to_evaluate,
            "context_count": count_context_in_answer,
            "bleu_score_mean": bleu_scores_mean,
            "bleu_score_median": bleu_scores_median,
            "cosine_similarity_mean": vector_similarity_mean,
            "cosine_similarity_median": vector_similarity_median,
        }
    )
