from asyncio.log import logger

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)

from app.memory import get_memory
from llama_index.core.prompts import PromptTemplate

from app.config import (
    get_embedding_model,
    get_llm
)


STORAGE_DIR = "./storage/vector_index"


SYSTEM_PROMPT = """
You are a legal information assistant specializing in Pakistani law.

Answer the user's question using the provided legal context.

Rules:

1. Explain legal provisions in clear English.
2. Do not invent statutory sections or court judgments.
3. Cite the relevant source document and page.
4. Distinguish actual legal text from your interpretation.
5. Mention important exceptions where supported by context.
6. If the context is insufficient, say so.
7. Do not present uncertain legal interpretations
   as established facts.

Context:

{context_str}

Question:

{query_str}

Answer:
"""

def create_query_engine(model_name, streaming=False):

    Settings.embed_model = get_embedding_model()

    llm = get_llm(model_name)

    from pathlib import Path
    # If the persisted index exists, load it; otherwise run ingestion to build it.
    if (Path(STORAGE_DIR) / "docstore.json").exists():
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    else:
        # No index yet – ingest PDFs from the data directory.
        from app.ingestion import ingest_documents
        index = ingest_documents()

    prompt = PromptTemplate(SYSTEM_PROMPT)

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=5,
        text_qa_template=prompt,
        streaming=streaming
    )

    return query_engine


def create_chat_engine(conversation_id, model_name, streaming=False):

    print(f"Creating chat engine for conversation_id=%s, model_name=%s", conversation_id, model_name)

    Settings.embed_model = get_embedding_model()

    llm = get_llm(model_name)

    from pathlib import Path
    # If the persisted index exists, load it; otherwise run ingestion to build it.
    if (Path(STORAGE_DIR) / "docstore.json").exists():
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        index = load_index_from_storage(storage_context)
    else:
        # No index yet – ingest PDFs from the data directory.
        from app.ingestion import ingest_documents
        index = ingest_documents()

    prompt = PromptTemplate(SYSTEM_PROMPT)

    memory = get_memory(conversation_id)

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        llm=llm,
        memory=memory,
        similarity_top_k=5,
        streaming=streaming,
        system_prompt="""
        You are a legal assistant specializing
        in Pakistani law.

        Answer in understandable English.

        Use retrieved legal documents as your
        primary source of legal authority.

        Use conversation history to understand
        follow-up questions and the user's situation.

        Never invent legal provisions or citations.
        Clearly identify uncertainty.
        """
    )

    return chat_engine

    # query_engine = index.as_query_engine(
    #     llm=llm,
    #     similarity_top_k=5,
    #     text_qa_template=prompt,
    #     streaming=streaming
    # )

    # return query_engine


def _extract_sources(response):
    sources = []

    for item in response.source_nodes:

        metadata = item.node.metadata

        sources.append({
            "file": metadata.get("file_name"),
            "page": metadata.get("page"),
            "score": item.score
        })

    return sources


def ask(question, model_name="paranine/gpt-oss-120b(high)"):

    engine = create_query_engine(model_name)

    response = engine.query(question)

    return {
        "answer": str(response),
        "sources": _extract_sources(response)
    }


def ask_stream(conversation_id, question, model_name="paranine/gpt-oss-120b(high)"):

    chat_engine = create_chat_engine(conversation_id=conversation_id, model_name=model_name, streaming=True)

    response = chat_engine.stream_chat(question)

    for token in response.response_gen:
        yield {
            "type": "token",
            "text": token
        }

    yield {
        "type": "sources",
        "sources": _extract_sources(response)
    }