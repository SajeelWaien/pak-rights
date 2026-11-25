import fitz
import shutil

from pathlib import Path

from llama_index.core import (
    Document,
    Settings,
    VectorStoreIndex
)

from llama_index.core.node_parser import SentenceSplitter

from app.config import get_embedding_model


# Resolve paths relative to the project root (the directory that contains the
# ``pak-rights`` folder). This works both when the code is run from the host
# (where the current working directory is the project root) and when it is
# executed inside Docker (where ``WORKDIR`` is ``/app``).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage" / "vector_index"


def load_pdfs():

    documents = []

    for pdf_path in DATA_DIR.rglob("*.pdf"):

        pdf = fitz.open(pdf_path)

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": pdf_path.name,
                        "page": page_number,
                        "document_type": pdf_path.parent.name
                    }
                )
            )

        pdf.close()

    return documents


def ingest_documents():

    Settings.embed_model = get_embedding_model()

    documents = load_pdfs()

    if not documents:
        raise ValueError("No extractable PDF text found.")

    splitter = SentenceSplitter(
        chunk_size=850,
        chunk_overlap=120
    )

    nodes = splitter.get_nodes_from_documents(documents)

    # Create a fresh index for this POC.
    shutil.rmtree(STORAGE_DIR, ignore_errors=True)

    index = VectorStoreIndex(
        nodes,
        show_progress=True
    )

    index.storage_context.persist(
        persist_dir=str(STORAGE_DIR)
    )

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(nodes)}")
    print("Index saved successfully.")

    return index


if __name__ == "__main__":
    ingest_documents()