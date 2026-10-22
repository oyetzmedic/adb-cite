"""Measure retrieval and abstention against a question set you wrote yourself.

questions.csv columns:
    id          any label
    question    the question text
    source      PDF file name the answer is in (leave blank to accept any file)
    gold_pages  page numbers where the answer is found, separated by ';'
    answerable  yes / no   (use "no" for deliberately out-of-scope questions)
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .answer import build_answer
from .retrieve import MODES, Index


@dataclass
class Question:
    qid: str
    question: str
    source: str
    gold_pages: set[int]
    answerable: bool


def load_questions(path: Path) -> list[Question]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    questions = []
    for row in rows:
        if not (row.get("question") or "").strip():
            continue
        pages = {int(p) for p in (row.get("gold_pages") or "").replace(",", ";").split(";") if p.strip()}
        answerable = (row.get("answerable") or "yes").strip().lower() in ("yes", "y", "true", "1")
        if answerable and not pages:
            raise ValueError(f"question {row.get('id')!r} is answerable but has no gold_pages")
        questions.append(Question((row.get("id") or "").strip(), row["question"].strip(),
                                  (row.get("source") or "").strip(), pages, answerable))
    return questions


def first_correct_rank(index: Index, q: Question, mode: str, depth: int = 10) -> int | None:
    for hit in index.search(q.question, mode=mode, k=depth):
        if hit.chunk.page in q.gold_pages and (not q.source or hit.chunk.source == q.source):
            return hit.rank
    return None


def retrieval_metrics(index: Index, questions: list[Question], mode: str) -> dict[str, float]:
    answerable = [q for q in questions if q.answerable]
    ranks = [first_correct_rank(index, q, mode) for q in answerable]
    n = len(answerable) or 1
    return {
        "n": len(answerable),
        "recall@1": sum(r == 1 for r in ranks) / n,
        "recall@5": sum(r is not None and r <= 5 for r in ranks) / n,
        "mrr@10": sum(1.0 / r for r in ranks if r) / n,
    }


def abstention_metrics(index: Index, questions: list[Question], min_similarity: float) -> dict[str, float]:
    answers = [(q, build_answer(q.question, index, min_similarity=min_similarity)) for q in questions]
    out_scope = [a for q, a in answers if not q.answerable]
    in_scope = [a for q, a in answers if q.answerable]
    return {
        "out_of_scope_n": len(out_scope),
        "out_of_scope_abstained": sum(a.abstained for a in out_scope),
        "in_scope_n": len(in_scope),
        "in_scope_wrongly_abstained": sum(a.abstained for a in in_scope),
    }


def run(index: Index, questions_path: Path, out_path: Path, min_similarity: float = 0.40) -> str:
    questions = load_questions(questions_path)
    lines = [
        "# ADB-Cite evaluation results", "",
        f"- Date: {date.today():%d/%m/%Y}",
        f"- Embedding model: {index.embedder.name}",
        f"- Chunks indexed: {len(index.chunks)}",
        f"- Questions: {len(questions)} ({sum(q.answerable for q in questions)} answerable, "
        f"{sum(not q.answerable for q in questions)} out of scope)",
        f"- Abstention threshold (cosine similarity): {min_similarity:.2f}", "",
        "## Retrieval (a hit counts when a retrieved chunk is on a gold page)", "",
        "| Method | Questions | Recall@1 | Recall@5 | MRR@10 |", "|---|---|---|---|---|",
    ]
    for mode in MODES:
        m = retrieval_metrics(index, questions, mode)
        lines.append(f"| {mode} | {m['n']} | {m['recall@1']:.2f} | {m['recall@5']:.2f} | {m['mrr@10']:.2f} |")
    a = abstention_metrics(index, questions, min_similarity)
    lines += [
        "", "## Abstention", "",
        f"- Out-of-scope questions correctly refused: {a['out_of_scope_abstained']} of {a['out_of_scope_n']}",
        f"- Answerable questions wrongly refused: {a['in_scope_wrongly_abstained']} of {a['in_scope_n']}",
        "", "## Best-passage similarity per question (use this to choose the threshold)", "",
        "| id | answerable | top similarity |", "|---|---|---|",
    ]
    for q in questions:
        lines.append(f"| {q.qid} | {'yes' if q.answerable else 'no'} | {float(index.dense_scores(q.question).max()):.2f} |")
    lines += ["", "Faithfulness of quoted answers is checked by hand; record that in the README.", ""]
    report = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return report
