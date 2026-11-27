import hashlib
import os
import re
import shutil
import tempfile
import uuid

from pathlib import Path

import pymupdf

from llama_index.core import (
    Document,
    Settings,
    VectorStoreIndex
)
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
    SentenceSplitter
)
from llama_index.core.schema import TextNode

from app.config import get_embedding_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage" / "vector_index"

SEMANTIC_SPLIT_MIN_TOKENS = 400
MAX_FALLBACK_CHUNK_TOKENS = 850

PART_RE = re.compile(
    r"^PART\s+([A-Z0-9IVXLCDM]+)\.?\s*(.*)$"
)
CHAPTER_RE = re.compile(
    r"^CHAPTER(?:\s*[-–]\s*|\s+)?"
    r"([A-Za-z0-9]+(?:\s*[-–]\s*[A-Za-z0-9]+)?)\.?\s*(.*)$"
)
SCHEDULE_RE = re.compile(
    r"^(?:\d+\[\s*)?(?:(FIRST|SECOND|THIRD|FOURTH|FIFTH)\s+)?"
    r"SCHEDULE(?![A-Z])\s*[-–]?\s*([A-Z0-9IVXLCDM]+)?"
    r"\s*[.\-–\]\[]?\s*(.*)$"
)
ANNEX_RE = re.compile(r"^(?:\d+\[\s*)?ANNEX\b(.*)$")
PROVISION_RE = re.compile(
    r"^(?:\d+\[\s*)?"
    r"(?P<number>\d{1,3}(?:\s*[-–]?\s*[A-Za-z]{1,2})?)"
    r"\s*\.\s*(?P<title>.*)$"
)


def _normalize_line(line: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        line.replace("\u00ad", "").replace("\u00a0", " ").replace("\u200b", "")
    ).strip()


def _find_body_start(pages: list[str]) -> tuple[int, int]:
    for page_index, page_text in enumerate(pages):
        lines = page_text.splitlines()
        for line_index, line in enumerate(lines):
            normalized = _normalize_line(line)
            if re.match(r"^preamble\b", normalized, re.IGNORECASE):
                return page_index, line_index

            if re.search(r"\bwhereas\b", normalized, re.IGNORECASE):
                following_text = "\n".join(lines[line_index:])
                if re.search(r"\benacted\b", following_text, re.IGNORECASE):
                    return page_index, line_index

    return 0, 0


def _parse_provision_heading(
    lines: list[str],
    line_index: int
) -> tuple[str, str] | None:
    line = _normalize_line(lines[line_index])
    match = PROVISION_RE.match(line)

    if match:
        number = re.sub(r"\s+", "", match.group("number")).replace("–", "-").upper()
        return number, match.group("title").strip()

    number_match = re.match(r"^(\d{1,3})\s*(?:[-–]\s*)?$", line)
    if number_match and line_index + 1 < len(lines):
        letter_match = re.match(
            r"^([A-Z]{1,2})\s*\.?\s*$",
            _normalize_line(lines[line_index + 1]),
            re.IGNORECASE
        )
        if letter_match:
            title = ""
            if line_index + 2 < len(lines):
                title = _normalize_line(lines[line_index + 2])
            return f"{number_match.group(1)}-{letter_match.group(1).upper()}", title

    return None


def _context_prefix(
    act_title: str,
    part_number: str | None,
    part_title: str | None,
    chapter_number: str | None,
    chapter_title: str | None,
    provision_type: str,
    provision_number: str,
    provision_title: str
) -> str:
    context = [act_title]
    if part_number:
        part = f"Part {part_number}"
        if part_title:
            part += f": {part_title}"
        context.append(part)
    if chapter_number:
        chapter = f"Chapter {chapter_number}"
        if chapter_title:
            chapter += f": {chapter_title}"
        context.append(chapter)

    if provision_type == "article":
        label = f"Article {provision_number}"
    elif provision_type == "section":
        label = f"Section {provision_number}"
    else:
        label = provision_number
    if provision_title:
        label += f": {provision_title}"
    context.append(label)
    return "\n".join(context)


def _parse_pdf_pages(pdf_path: Path, pages: list[str]) -> list[Document]:
    start_page, start_line = _find_body_start(pages)
    provision_type = (
        "article"
        if "constitution" in pdf_path.stem.lower()
        else "section"
    )
    source_lines = []

    for page_index in range(start_page, len(pages)):
        lines = pages[page_index].splitlines()
        line_start = start_line if page_index == start_page else 0
        for line in lines[line_start:]:
            normalized = _normalize_line(line)
            if normalized:
                source_lines.append((normalized, page_index + 1))
    source_text_lines = [line for line, _ in source_lines]

    documents = []
    current = None
    part_number = None
    part_title = None
    chapter_number = None
    chapter_title = None
    pending_heading = None
    in_schedule = False

    def flush_current():
        nonlocal current
        if current is None:
            return

        body = "\n".join(line for line, _ in current["lines"]).strip()
        if body:
            page_numbers = [page for _, page in current["lines"]]
            sequence = len(documents) + 1
            try:
                source_path = pdf_path.relative_to(DATA_DIR).as_posix()
            except ValueError:
                source_path = pdf_path.as_posix()
            provision_id = hashlib.sha256(
                f"{source_path}|{current['number']}|{min(page_numbers)}|{sequence}".encode(
                    "utf-8"
                )
            ).hexdigest()[:20]
            prefix = _context_prefix(
                pdf_path.stem,
                part_number,
                part_title,
                chapter_number,
                chapter_title,
                current["type"],
                current["number"],
                current["title"]
            )
            metadata = {
                "file_name": pdf_path.name,
                "source_path": source_path,
                "act_title": pdf_path.stem,
                "document_type": (
                    "act" if pdf_path.parent == DATA_DIR else pdf_path.parent.name
                ),
                "provision_type": current["type"],
                "provision_number": current["number"],
                "provision_id": provision_id,
                "page": min(page_numbers),
                "page_start": min(page_numbers),
                "page_end": max(page_numbers)
            }
            optional_metadata = {
                "part_number": part_number,
                "part_title": part_title,
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
                "provision_title": current["title"],
                "parent_provision_id": provision_id
            }
            metadata.update({
                key: value
                for key, value in optional_metadata.items()
                if value
            })
            documents.append(
                Document(
                    id_=provision_id,
                    text=f"{prefix}\n\n{body}",
                    metadata=metadata
                )
            )

        current = None

    for line_index, (line, page_number) in enumerate(source_lines):
        part_match = PART_RE.match(line)
        if part_match:
            if in_schedule:
                current["lines"].append((line, page_number))
                continue
            flush_current()
            part_number = part_match.group(1).upper()
            part_title = part_match.group(2).strip() or None
            chapter_number = None
            chapter_title = None
            pending_heading = "part"
            in_schedule = False
            continue

        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            if in_schedule:
                current["lines"].append((line, page_number))
                continue
            flush_current()
            chapter_number = chapter_match.group(1).upper()
            chapter_title = chapter_match.group(2).strip() or None
            pending_heading = "chapter"
            in_schedule = False
            continue

        schedule_match = SCHEDULE_RE.match(line)
        if schedule_match:
            flush_current()
            schedule_number = (
                schedule_match.group(1)
                or schedule_match.group(2)
                or "Unnumbered"
            )
            schedule_title = schedule_match.group(3).strip(" .-–[]")
            current = {
                "type": "schedule",
                "number": f"Schedule {schedule_number.upper()}",
                "title": schedule_title,
                "lines": [(line, page_number)]
            }
            pending_heading = None
            in_schedule = True
            continue

        annex_match = ANNEX_RE.match(line)
        if annex_match:
            flush_current()
            current = {
                "type": "annex",
                "number": "Annex",
                "title": annex_match.group(1).strip(" .-–[]"),
                "lines": [(line, page_number)]
            }
            pending_heading = None
            in_schedule = True
            continue

        if pending_heading:
            if not _parse_provision_heading(
                source_text_lines,
                line_index
            ):
                if pending_heading == "part":
                    part_title = " ".join(
                        value for value in (part_title, line) if value
                    )
                else:
                    chapter_title = " ".join(
                        value for value in (chapter_title, line) if value
                    )
                continue
            pending_heading = None

        if not in_schedule:
            provision = _parse_provision_heading(
                source_text_lines,
                line_index
            )
            if provision:
                flush_current()
                provision_number, provision_title = provision
                current = {
                    "type": provision_type,
                    "number": provision_number,
                    "title": provision_title,
                    "lines": [(line, page_number)]
                }
                continue

        if current is None:
            current = {
                "type": "preamble",
                "number": "Preamble",
                "title": "Preamble",
                "lines": []
            }
        current["lines"].append((line, page_number))

    flush_current()
    return documents


def load_pdfs() -> list[Document]:
    documents = []
    pdf_paths = sorted(DATA_DIR.rglob("*.pdf"))
    print(
        f"PDF discovery complete: found {len(pdf_paths)} PDF(s) under {DATA_DIR}",
        flush=True,
    )

    for pdf_index, pdf_path in enumerate(pdf_paths, start=1):
        print(
            f"PDF [{pdf_index}/{len(pdf_paths)}] starting extraction: {pdf_path.name}",
            flush=True,
        )
        with pymupdf.open(pdf_path) as pdf:
            pages = [page.get_text("text") for page in pdf]
            print(
                f"PDF [{pdf_index}/{len(pdf_paths)}] extracted {len(pages)} page(s): {pdf_path.name}",
                flush=True,
            )
        parsed_documents = _parse_pdf_pages(pdf_path, pages)
        documents.extend(parsed_documents)
        print(
            f"PDF [{pdf_index}/{len(pdf_paths)}] parsed {len(parsed_documents)} "
            f"legal provision(s): {pdf_path.name}",
            flush=True,
        )

    print(
        f"PDF parsing checkpoint: extracted {len(documents)} legal provision(s) "
        f"from {len(pdf_paths)} PDF(s)",
        flush=True,
    )
    return documents


def _token_count(text: str) -> int:
    return len(Settings.tokenizer(text))


def _node_id(provision_id: str, chunk_index: int) -> str:
    return hashlib.sha256(
        f"{provision_id}|{chunk_index}".encode("utf-8")
    ).hexdigest()


def _build_nodes(documents: list[Document]) -> list[TextNode]:
    print(
        f"Chunking checkpoint: starting {len(documents)} legal provision(s)",
        flush=True,
    )
    semantic_splitter = SemanticSplitterNodeParser(
        embed_model=Settings.embed_model,
        breakpoint_percentile_threshold=95
    )
    safety_splitter = SentenceSplitter(
        chunk_size=MAX_FALLBACK_CHUNK_TOKENS,
        chunk_overlap=0
    )
    nodes = []

    for document_index, document in enumerate(documents, start=1):
        token_count = _token_count(document.text)
        if token_count > SEMANTIC_SPLIT_MIN_TOKENS:
            print(
                f"Semantic split [{document_index}/{len(documents)}]: "
                f"{document.metadata.get('file_name')} "
                f"{document.metadata.get('provision_number')} ({token_count} tokens)",
                flush=True,
            )
            semantic_nodes = semantic_splitter.get_nodes_from_documents([document])
            chunks = []
            for semantic_node in semantic_nodes:
                if _token_count(semantic_node.text) > MAX_FALLBACK_CHUNK_TOKENS:
                    chunks.extend(
                        (text, "token_safety")
                        for text in safety_splitter.split_text(semantic_node.text)
                    )
                else:
                    chunks.append((semantic_node.text, "semantic"))
        else:
            chunks = [(document.text, "provision")]

        for chunk_index, (text, strategy) in enumerate(chunks):
            metadata = dict(document.metadata)
            metadata["chunk_index"] = chunk_index
            metadata["chunking_strategy"] = strategy
            nodes.append(
                TextNode(
                    id_=_node_id(document.metadata["provision_id"], chunk_index),
                    text=text,
                    metadata=metadata
                )
            )

        if document_index % 100 == 0 or document_index == len(documents):
            print(
                f"Chunking checkpoint: processed {document_index}/{len(documents)} "
                f"provisions; generated {len(nodes)} chunk(s)",
                flush=True,
            )

    print(
        f"Chunking complete: generated {len(nodes)} chunk(s) "
        f"from {len(documents)} provision(s)",
        flush=True,
    )
    return nodes


def _persist_index(index: VectorStoreIndex) -> None:
    STORAGE_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Persistence checkpoint: creating temporary index at {STORAGE_DIR.parent}",
        flush=True,
    )
    temporary_dir = Path(tempfile.mkdtemp(
        prefix=f".{STORAGE_DIR.name}.new-",
        dir=STORAGE_DIR.parent
    ))
    print(
        f"Persistence checkpoint: temporary directory ready: {temporary_dir}",
        flush=True,
    )
    backup_dir = None

    try:
        print("Persistence checkpoint: writing index files", flush=True)
        index.storage_context.persist(persist_dir=str(temporary_dir))
        print("Persistence checkpoint: temporary index write complete", flush=True)
        if STORAGE_DIR.exists():
            backup_dir = STORAGE_DIR.with_name(
                f"{STORAGE_DIR.name}.backup-{uuid.uuid4().hex}"
            )
            print(
                f"Persistence checkpoint: backing up existing index to {backup_dir}",
                flush=True,
            )
            os.replace(STORAGE_DIR, backup_dir)

        try:
            os.replace(temporary_dir, STORAGE_DIR)
            print(
                f"Persistence checkpoint: new index installed at {STORAGE_DIR}",
                flush=True,
            )
        except OSError:
            if backup_dir and backup_dir.exists():
                os.replace(backup_dir, STORAGE_DIR)
                print(
                    "Persistence checkpoint: restored existing index after install failure",
                    flush=True,
                )
            raise

        if backup_dir:
            shutil.rmtree(backup_dir)
            print("Persistence checkpoint: removed old index backup", flush=True)
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
            print("Persistence checkpoint: cleaned temporary directory", flush=True)


def ingest_documents():
    print("Ingestion checkpoint: loading embedding model", flush=True)
    Settings.embed_model = get_embedding_model()
    print("Ingestion checkpoint: embedding model ready", flush=True)

    documents = load_pdfs()
    if not documents:
        raise ValueError("No extractable legal PDF text found.")

    nodes = _build_nodes(documents)
    if not nodes:
        raise ValueError("No legal chunks were generated from the source PDFs.")

    print(
        f"Ingestion checkpoint: starting vector index construction "
        f"for {len(nodes)} chunk(s)",
        flush=True,
    )
    index = VectorStoreIndex(
        nodes,
        show_progress=True
    )
    print("Ingestion checkpoint: vector index construction complete", flush=True)
    _persist_index(index)

    semantic_count = sum(
        node.metadata.get("chunking_strategy") == "semantic"
        for node in nodes
    )
    fallback_count = sum(
        node.metadata.get("chunking_strategy") == "token_safety"
        for node in nodes
    )
    print(f"Ingestion summary: legal provisions={len(documents)}", flush=True)
    print(f"Ingestion summary: chunks={len(nodes)}", flush=True)
    print(f"Ingestion summary: semantic_chunks={semantic_count}", flush=True)
    print(f"Ingestion summary: token_safety_chunks={fallback_count}", flush=True)
    print("Ingestion complete: index saved successfully", flush=True)

    return index


if __name__ == "__main__":
    ingest_documents()
