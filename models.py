import os

os.environ["HF_HOME"] = "./hf_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./hf_cache"
os.environ["HF_HUB_OFFLINE"] = "0"  # switch to "1" after first download

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

HF_TOKEN = "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"
EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v5-text-nano"
RERANKING_MODEL_ID = "BAAI/bge-reranker-v2-m3"
EMBEDDING_MODEL_PATH = "./hf_cache/embedding_model"
RERANKING_MODEL_PATH = "./hf_cache/reranking_model"

embedding_model = None
re_ranking_model = None


def ensure_models_downloaded():
    """Run once to download models. Skips if already present."""
    os.makedirs("./hf_cache", exist_ok=True)

    if not os.path.exists(EMBEDDING_MODEL_PATH):
        print(f"Downloading embedding model: {EMBEDDING_MODEL_ID}")
        snapshot_download(
            repo_id=EMBEDDING_MODEL_ID,
            repo_type="model",
            local_dir=EMBEDDING_MODEL_PATH,
            revision="refs/pr/11",
            token=HF_TOKEN,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],  # skip unused formats
        )
        print(f"Embedding model saved to {EMBEDDING_MODEL_PATH}")
    else:
        print("Embedding model already downloaded, skipping.")

    if not os.path.exists(RERANKING_MODEL_PATH):
        print(f"Downloading reranking model: {RERANKING_MODEL_ID}")
        snapshot_download(
            repo_id=RERANKING_MODEL_ID,
            repo_type="model",
            local_dir=RERANKING_MODEL_PATH,
            token=HF_TOKEN,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],  # skip unused formats
        )
        print(f"Reranking model saved to {RERANKING_MODEL_PATH}")
    else:
        print("Reranking model already downloaded, skipping.")


def init_models():
    global embedding_model, re_ranking_model

    ensure_models_downloaded()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing models on {device}...")

    try:
        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_PATH,
            trust_remote_code=True,
            device=device,
            model_kwargs={
                "dtype": torch.bfloat16,
                "default_task": "retrieval"
            },
            local_files_only=True,
        )
        print("Embedding model loaded.")
    except Exception as e:
        print(f"Embedding model not found. Run ensure_models_downloaded() first.\n{e}")
        raise

    try:
        re_ranking_model = CrossEncoder(
            RERANKING_MODEL_PATH,
            device=device,
            local_files_only=True,
        )
        print("Reranking model loaded.")
    except Exception as e:
        print(f"Reranking model not found. Run ensure_models_downloaded() first.\n{e}")
        raise

    print(f"All models ready on {device}.")


def get_embedding_model() -> SentenceTransformer:
    global embedding_model
    if embedding_model is None:
        init_models()
    return embedding_model


def get_reranking_model() -> CrossEncoder:
    global re_ranking_model
    if re_ranking_model is None:
        init_models()
    return re_ranking_model