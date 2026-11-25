from llama_index.core.memory import Memory

DATABASE_URL = "sqlite+aiosqlite:///./storage/chat_history.db"


def get_memory(conversation_id: str):

    return Memory.from_defaults(
        session_id=conversation_id,

        # Maximum short-term + long-term memory budget
        token_limit=6000,

        # Keep most of the budget for recent chat history
        chat_history_token_ratio=0.7,

        token_flush_size=500,

        async_database_uri=DATABASE_URL,
        table_name="legal_chat_memory"
    )