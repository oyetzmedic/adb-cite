"""Build an answer made only of sentences quoted from the source, each with a citation.

There is no text generation here. The tool either quotes the document or
abstains. That is deliberate: in a safety-critical setting an unsupported
answer is worse than no answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .retrieve import Hit, Index

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
DISCLAIMER = "Quoted from the source text. Not engineering advice; check the current edition."


@dataclass
class CitedSentence:
    text: str
    source: str
    page: int
    para: str | None
    similarity: float

    def citation(self) -> str:
        para = f", para {self.para}" if self.para else ""
        return f"[{self.source}, p. {self.page}{para}]"


@dataclass
class Answer:
    question: str
    abstained: bool
    top_similarity: float
    reason: str = ""
    sentences: list[CitedSentence] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)


def split_sentences(text: str, min_words: int = 6) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if len(s.split()) >= min_words]


def build_answer(question: str, index: Index, mode: str = "hybrid", k: int = 5,
                 max_sentences: int = 3, min_similarity: float = 0.40,
                 sentence_ratio: float = 0.6) -> Answer:
    """Retrieve passages, then quote the sentences closest to the question.

    Abstains when the best passage's cosine similarity to the question is below
    ``min_similarity``. Tune that threshold on your own out-of-scope questions.
    A sentence is only quoted if it scores at least ``sentence_ratio`` times the
    best sentence, so weakly related sentences are left out.
    """
    top_similarity = float(index.dense_scores(question).max())
    hits = index.search(question, mode=mode, k=k)
    if top_similarity < min_similarity:
        return Answer(question, True, top_similarity,
                      reason=f"best passage similarity {top_similarity:.2f} is below {min_similarity:.2f}",
                      hits=hits)

    candidates = [(sentence, hit) for hit in hits for sentence in split_sentences(hit.chunk.text)]
    if not candidates:
        return Answer(question, True, top_similarity, reason="no quotable sentence found", hits=hits)

    query_vector = index.embedder.encode([question])[0]
    similarities = index.embedder.encode([s for s, _ in candidates]) @ query_vector
    ranked = sorted(zip(similarities, candidates), key=lambda pair: -pair[0])[:max_sentences]
    best = float(ranked[0][0])
    ranked = [pair for pair in ranked if float(pair[0]) >= sentence_ratio * best]
    sentences = [CitedSentence(s, h.chunk.source, h.chunk.page, h.chunk.para, float(sim))
                 for sim, (s, h) in ranked]
    return Answer(question, False, top_similarity, sentences=sentences, hits=hits)


def format_answer(answer: Answer) -> str:
    lines = [f"Q: {answer.question}"]
    if answer.abstained:
        lines.append(f"No supported answer found in the indexed documents ({answer.reason}).")
    else:
        for n, s in enumerate(answer.sentences, start=1):
            lines.append(f'{n}. "{s.text}" {s.citation()}')
        lines.append(DISCLAIMER)
    return "\n".join(lines)
