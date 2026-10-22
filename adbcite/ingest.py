"""Turn PDF pages into small text chunks that remember where they came from.

Every chunk keeps its source file, page number and (where one can be found)
the paragraph number printed in the document, e.g. "3.12". Those three fields
are what make source attribution possible later.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

PARA_NUMBER = re.compile(r"^\s*(\d{1,2}\.\d{1,3})\b")


@dataclass
class Chunk:
    chunk_id: str
    source: str          # PDF file name
    page: int            # 1-based page number in the PDF
    para: str | None     # first printed paragraph number found in the chunk
    text: str


def clean(text: str) -> str:
    """Collapse whitespace and repair words hyphenated across line breaks."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return " ".join(text.split())


def extract_blocks(pdf_path: Path) -> list[tuple[int, list[str]]]:
    """Return [(page_number, [block_text, ...]), ...] for a PDF."""
    import pymupdf  # imported here so the rest of the package works without it

    pages = []
    with pymupdf.open(pdf_path) as doc:
        for number, page in enumerate(doc, start=1):
            blocks = [clean(b[4]) for b in page.get_text("blocks") if b[6] == 0]
            pages.append((number, [b for b in blocks if b]))
    return pages


def chunk_blocks(source: str, page: int, blocks: list[str],
                 max_words: int = 180, min_words: int = 15) -> list[Chunk]:
    """Group a page's text blocks into chunks of at most ``max_words`` words.

    Chunks never cross a page boundary, so a citation always points at one page.
    """
    chunks: list[Chunk] = []
    current: list[str] = []
    count = 0

    def flush() -> None:
        nonlocal current, count
        text = " ".join(current).strip()
        if len(text.split()) >= min_words:
            match = next((PARA_NUMBER.match(b) for b in current if PARA_NUMBER.match(b)), None)
            chunks.append(Chunk(
                chunk_id=f"{Path(source).stem}-p{page}-c{len(chunks) + 1}",
                source=source, page=page,
                para=match.group(1) if match else None, text=text))
        current, count = [], 0

    for block in blocks:
        words = block.split()
        # a single very long block is split on its own
        while len(words) > max_words:
            if current:
                flush()
            current, count = [" ".join(words[:max_words])], max_words
            flush()
            words = words[max_words:]
        if count + len(words) > max_words and current:
            flush()
        if words:
            current.append(" ".join(words))
            count += len(words)
    flush()
    return chunks


def ingest_folder(pdf_dir: Path, max_words: int = 180) -> list[Chunk]:
    chunks: list[Chunk] = []
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        for page, blocks in extract_blocks(pdf):
            chunks.extend(chunk_blocks(pdf.name, page, blocks, max_words=max_words))
    return chunks


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]
