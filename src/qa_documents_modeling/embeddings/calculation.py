from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "paraphrase-multilingual-mpnet-base-v2", cache_folder="assets"
)


def get_embeddings(sentences: list[str]):
    return model.encode(sentences)
