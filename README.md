[![Tests](https://github.com/oyetzmedic/adb-cite/actions/workflows/tests.yml/badge.svg)](https://github.com/oyetzmedic/adb-cite/actions/workflows/tests.yml)

# ADB-Cite: source-attributed retrieval over Approved Document B

A small, self-directed prototype. You ask a question about fire-safety guidance; the tool finds the most relevant passages in Approved Document B and answers only with sentences quoted from the document, each with its file, page and paragraph. If nothing relevant is found, it says so instead of answering.

It does not generate text. That is deliberate. In a safety-critical setting an unsupported answer is worse than no answer, so this version either quotes the source or abstains.

**Status:** learning project, October 2026. Not validated by fire engineers. Not for design decisions.

## What it does

1. **Ingest.** Reads each PDF page with PyMuPDF and groups the text into chunks of up to 180 words. A chunk never crosses a page, so every citation points at one page. Where a printed paragraph number (for example 3.12) starts a block, it is kept.
2. **Index.** Builds two indexes over the chunks: BM25 keyword search (`rank-bm25`) and dense embeddings (`sentence-transformers`, model `all-MiniLM-L6-v2`), searched by exact cosine similarity in NumPy.
3. **Retrieve.** Three methods: BM25, dense, and a hybrid that merges the two rankings by reciprocal rank fusion.
4. **Answer with citations.** Splits the top passages into sentences and quotes the ones closest to the question, each followed by `[file, p. N, para X.Y]`. Weakly related sentences are left out.
5. **Abstain.** If the best passage's similarity to the question is below a threshold, the tool returns "No supported answer found".
6. **Evaluate.** Scores each retrieval method against a question set with known answer pages, and counts how often out-of-scope questions are correctly refused.

## Set-up (Windows, Anaconda)

```
conda create -n adbcite python=3.11 -y
conda activate adbcite
pip install -r requirements.txt
```

Download Approved Document B (Volume 1 and Volume 2) from GOV.UK and put the PDFs in `data/`.

## Use

```
python -m adbcite build
python -m adbcite ask "What is the maximum travel distance in a flat?"
python -m adbcite ask "..." --mode bm25
python -m adbcite eval
pytest -q
```

`build` downloads the embedding model the first time it runs (about 90 MB).

## Evaluation

### The question set

`eval/questions.csv` has one row per question:

| Column | Meaning |
|---|---|
| `id` | any label, for example `q01` |
| `question` | the question |
| `source` | the PDF file name, or blank to accept either volume |
| `gold_pages` | PDF page number(s) where the answer is, separated by `;` |
| `answerable` | `yes`, or `no` for a deliberately out-of-scope question |

I wrote the questions by reading the document and noting the page each answer is on. Page numbers are PDF page numbers (the first page of the file is 1), not the numbers printed on the page.

### What is measured

- **Recall@1 and Recall@5:** the share of answerable questions where a chunk from a gold page is ranked first, or in the top five.
- **MRR@10:** mean reciprocal rank of the first correct chunk within the top ten.
- **Abstention:** how many out-of-scope questions are refused, and how many answerable questions are wrongly refused.
- **Faithfulness (by hand):** for a sample of answers, whether each quoted sentence actually answers the question.

### Results

`python -m adbcite eval` writes the retrieval and abstention figures to `eval/results.md`.

> On a 35-question evaluation set containing 30 answerable questions and 5 deliberately out-of-scope questions, BM25 achieved Recall@1 of 0.70, Recall@5 of 0.97 and MRR@10 of 0.83. Hybrid retrieval achieved 0.60, 0.97 and 0.73 respectively, while dense retrieval achieved 0.30, 0.73 and 0.49.

Using an abstention threshold of 0.40, the system correctly refused all 5 out-of-scope questions and wrongly refused 0 of 30 answerable questions. In this small test set, the lowest best-passage similarity among answerable questions was 0.56 and the highest among out-of-scope questions was 0.36.

For the manual faithfulness check, I reviewed 15 quoted answers. None had every returned quote directly answering the question, 10 were partly supported because at least one returned quote directly answered the question, and 5 were not supported because none of the returned quotes directly answered the question. This exposed a limitation in the sentence-selection stage, particularly for short numerical or list-item answers, even when the correct source page was retrieved.

## Limitations

- The question set and gold pages were written by me. I am not a fire engineer, so the labels have not been checked by an expert.
- The test set is small. The figures describe this test set and should not be read as general performance.
- A hit is counted at page level. The right page can be retrieved without the right sentence being quoted.
- Answers are quoted sentences only. The tool does not reason across passages, compare clauses or resolve conflicts between them.
- Tables and diagrams in the PDF are not handled properly; text extracted from tables can be jumbled.
- The abstention threshold was chosen on a handful of out-of-scope questions and may not hold for others.
- Guidance is amended over time. The index reflects only the edition of the PDF that was downloaded.
- British Standards and other copyright documents are not included and should not be added.

## What I would do next

- Have a qualified engineer write and check a larger question set.
- Add a re-ranking step and compare it with the hybrid method.
- Add a generated summary on top of the quoted evidence, with an automatic check that every statement is supported by a cited passage.
- Read documents from a managed library such as SharePoint, keeping each user's access permissions.

## Source and licence of the guidance

Approved Document B is published by the UK Government on GOV.UK. Check the licence statement on the copyright page of the PDF you download and follow its attribution requirement. The PDFs are not stored in this repository.

## How this was built

I built this as a learning project with AI coding assistance. The question set, the gold labels, the evaluation run and the manual checks are my own.
