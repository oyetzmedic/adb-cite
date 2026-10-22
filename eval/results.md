# ADB-Cite evaluation results

- Date: 21/10/2026
- Embedding model: all-MiniLM-L6-v2
- Chunks indexed: 1097
- Questions: 35 (30 answerable, 5 out of scope)
- Abstention threshold (cosine similarity): 0.40

## Retrieval (a hit counts when a retrieved chunk is on a gold page)

| Method | Questions | Recall@1 | Recall@5 | MRR@10 |
|---|---|---|---|---|
| bm25 | 30 | 0.70 | 0.97 | 0.83 |
| dense | 30 | 0.30 | 0.73 | 0.49 |
| hybrid | 30 | 0.60 | 0.97 | 0.73 |

## Abstention

- Out-of-scope questions correctly refused: 5 of 5
- Answerable questions wrongly refused: 0 of 30

## Best-passage similarity per question (use this to choose the threshold)

| id | answerable | top similarity |
|---|---|---|
| q01 | yes | 0.77 |
| q02 | yes | 0.74 |
| q03 | yes | 0.74 |
| q04 | yes | 0.69 |
| q05 | yes | 0.73 |
| q06 | yes | 0.78 |
| q07 | yes | 0.64 |
| q08 | yes | 0.62 |
| q09 | yes | 0.57 |
| q10 | yes | 0.67 |
| q11 | yes | 0.66 |
| q12 | yes | 0.80 |
| q13 | yes | 0.61 |
| q14 | yes | 0.64 |
| q15 | yes | 0.68 |
| q16 | yes | 0.72 |
| q17 | yes | 0.75 |
| q18 | yes | 0.67 |
| q19 | yes | 0.68 |
| q20 | yes | 0.58 |
| q21 | yes | 0.76 |
| q22 | yes | 0.73 |
| q23 | yes | 0.61 |
| q24 | yes | 0.56 |
| q25 | yes | 0.75 |
| q26 | yes | 0.63 |
| q27 | yes | 0.63 |
| q28 | yes | 0.70 |
| q29 | yes | 0.69 |
| q30 | yes | 0.76 |
| q31 | no | 0.36 |
| q32 | no | 0.27 |
| q33 | no | 0.33 |
| q34 | no | 0.24 |
| q35 | no | 0.26 |

Faithfulness of quoted answers is checked by hand; record that in the README.
