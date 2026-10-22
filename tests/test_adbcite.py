"""Unit tests. They use the hashing stand-in embedder, so no model download is needed."""
from pathlib import Path

import pytest

from adbcite.answer import build_answer, format_answer, split_sentences
from adbcite.embed import HashingEmbedder, tokenize
from adbcite.evaluate import load_questions, retrieval_metrics, run
from adbcite.ingest import Chunk, chunk_blocks, extract_blocks, load_chunks, save_chunks
from adbcite.retrieve import Index

TEXTS = {
    3: "2.1 Every dwelling should have a smoke alarm on each storey. The alarms should be "
       "mains powered with a standby supply and linked so that all sound together.",
    7: "3.4 An escape window should have an unobstructed openable area of at least 0.33 square "
       "metres. The bottom of the openable area should be no more than 1100 mm above the floor.",
    12: "5.2 Cavity barriers should be provided at the edges of cavities and around openings. "
        "They restrict the spread of smoke and flame within concealed spaces in the wall.",
}


@pytest.fixture()
def index():
    chunks = [Chunk(f"demo-p{p}-c1", "demo.pdf", p, t.split()[0], t) for p, t in TEXTS.items()]
    return Index.build(chunks, HashingEmbedder())


def test_tokenize_keeps_paragraph_numbers():
    assert "3.4" in tokenize("See paragraph 3.4 for windows")


def test_chunks_stay_on_one_page_and_keep_paragraph_number():
    blocks = ["3.4 " + "word " * 100, "more " * 100, "tail " * 30]
    chunks = chunk_blocks("demo.pdf", 7, blocks, max_words=120)
    assert len(chunks) >= 2
    assert all(c.page == 7 and len(c.text.split()) <= 120 for c in chunks)
    assert chunks[0].para == "3.4"


def test_very_long_block_is_split():
    chunks = chunk_blocks("demo.pdf", 1, ["x " * 500], max_words=180)
    assert len(chunks) == 3 and all(len(c.text.split()) <= 180 for c in chunks)


def test_short_fragments_are_dropped():
    assert chunk_blocks("demo.pdf", 1, ["Page 4"]) == []


def test_chunks_round_trip(tmp_path):
    chunks = chunk_blocks("demo.pdf", 2, [TEXTS[3]])
    save_chunks(chunks, tmp_path / "chunks.jsonl")
    assert load_chunks(tmp_path / "chunks.jsonl") == chunks


def test_pdf_extraction_reports_page_numbers(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for text in ("First page about smoke alarms in dwellings.", "Second page about escape windows."):
        doc.new_page().insert_text((72, 72), text)
    path = tmp_path / "two_pages.pdf"
    doc.save(path)
    pages = extract_blocks(path)
    assert [n for n, _ in pages] == [1, 2] and "escape windows" in pages[1][1][0]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_each_method_finds_the_right_page(index, mode):
    hits = index.search("How large should an escape window be?", mode=mode, k=3)
    assert hits[0].chunk.page == 7 and hits[0].rank == 1


def test_unknown_mode_is_rejected(index):
    with pytest.raises(ValueError):
        index.search("anything", mode="magic")


def test_index_round_trip(index, tmp_path):
    index.save(tmp_path)
    again = Index.load(tmp_path, HashingEmbedder())
    assert again.search("smoke alarm storey", k=1)[0].chunk.page == 3


def test_answer_quotes_the_source_with_a_citation(index):
    answer = build_answer("What openable area should an escape window have?", index, min_similarity=0.2)
    assert not answer.abstained
    assert answer.sentences[0].text in TEXTS[7]
    text = format_answer(answer)
    assert "[demo.pdf, p. 7, para 3.4]" in text and "Not engineering advice" in text


def test_weakly_related_sentences_are_not_quoted(index):
    answer = build_answer("What openable area should an escape window have?", index, min_similarity=0.2)
    assert all(s.page == 7 for s in answer.sentences)


def test_answer_abstains_when_nothing_is_relevant(index):
    answer = build_answer("Which football club won the league?", index, min_similarity=0.2)
    assert answer.abstained and "No supported answer" in format_answer(answer)


def test_split_sentences_drops_fragments():
    assert split_sentences("Short one. This sentence is long enough to be kept as a quote.") == [
        "This sentence is long enough to be kept as a quote."]


def test_evaluation_metrics_and_report(index, tmp_path):
    csv_path = tmp_path / "questions.csv"
    csv_path.write_text(
        "id,question,source,gold_pages,answerable\n"
        "q1,How large should an escape window be?,demo.pdf,7,yes\n"
        "q2,Where are cavity barriers needed?,,12,yes\n"
        "q3,Which football club won the league?,,,no\n", encoding="utf-8")
    questions = load_questions(csv_path)
    assert len(questions) == 3 and questions[2].answerable is False
    metrics = retrieval_metrics(index, questions, "bm25")
    assert metrics["n"] == 2 and metrics["recall@1"] == 1.0 and metrics["mrr@10"] == 1.0
    report = run(index, csv_path, tmp_path / "results.md", min_similarity=0.2)
    assert "| hybrid | 2 |" in report and "correctly refused: 1 of 1" in report
    assert (tmp_path / "results.md").exists()


def test_answerable_question_without_gold_page_is_an_error(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,question,source,gold_pages,answerable\nq1,Anything?,,,yes\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_questions(path)
