import os
from dotenv import load_dotenv

from llama_index.embeddings.openai_like import (
    OpenAILikeEmbedding
)

from llama_index.llms.openai_like import OpenAILike

load_dotenv()

API_KEY = os.getenv("P9_API_KEY")  # Load from .env or environment; keep None if not set

BASE_URL = "https://sovereign.paranine.com/v1"

EMBEDDING_MODEL = "paranine/qwen3-embedding"

CHAT_MODELS = {
    "paranine/gpt-oss-120b(high)": 128000,
    "paranine/qwen-3.8-27b": 32768,
}


def get_embedding_model():

    return OpenAILikeEmbedding(
        model_name=EMBEDDING_MODEL,
        api_base=BASE_URL,
        api_key=API_KEY,
        embed_batch_size=10
    )


def get_llm(model_name="paranine/gpt-oss-120b(high)"):

    context_window = CHAT_MODELS[model_name]

    return OpenAILike(
        model=model_name,
        api_base=BASE_URL,
        api_key=API_KEY,
        is_chat_model=True,
        context_window=context_window
    )