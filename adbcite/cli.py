"""Command line: build the index, ask a question, run the evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .answer import build_answer, format_answer
from .embed import get_embedder
from .evaluate import run as run_evaluation
from .ingest import ingest_folder
from .retrieve import MODES, Index


def _load(index_dir: Path) -> Index:
    meta = json.loads((index_dir / "meta.json").read_text())
    return Index.load(index_dir, get_embedder(meta["embedder"]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="adbcite", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="read the PDFs in --pdf-dir and build the index")
    b.add_argument("--pdf-dir", type=Path, default=Path("data"))
    b.add_argument("--index-dir", type=Path, default=Path("index"))
    b.add_argument("--embedder", default="all-MiniLM-L6-v2")
    b.add_argument("--max-words", type=int, default=180)

    a = sub.add_parser("ask", help="ask a question and get quoted, cited sentences")
    a.add_argument("question")
    a.add_argument("--index-dir", type=Path, default=Path("index"))
    a.add_argument("--mode", choices=MODES, default="hybrid")
    a.add_argument("--min-similarity", type=float, default=0.40)

    e = sub.add_parser("eval", help="score retrieval and abstention on your question set")
    e.add_argument("--questions", type=Path, default=Path("eval/questions.csv"))
    e.add_argument("--out", type=Path, default=Path("eval/results.md"))
    e.add_argument("--index-dir", type=Path, default=Path("index"))
    e.add_argument("--min-similarity", type=float, default=0.40)

    args = parser.parse_args(argv)
    if args.command == "build":
        chunks = ingest_folder(args.pdf_dir, max_words=args.max_words)
        if not chunks:
            raise SystemExit(f"No text found. Put the PDFs in {args.pdf_dir}/ first.")
        Index.build(chunks, get_embedder(args.embedder)).save(args.index_dir)
        print(f"Indexed {len(chunks)} chunks into {args.index_dir}/")
    elif args.command == "ask":
        print(format_answer(build_answer(args.question, _load(args.index_dir), mode=args.mode,
                                         min_similarity=args.min_similarity)))
    else:
        print(run_evaluation(_load(args.index_dir), args.questions, args.out, args.min_similarity))


if __name__ == "__main__":
    main()
