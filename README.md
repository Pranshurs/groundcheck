# GroundCheck

Grounding verification for RAG outputs. Given a source text and an answer, GroundCheck
predicts whether the answer is actually supported by the source, with a confidence score.
It catches the common failure modes of RAG systems: contradicted facts, swapped numbers,
and claims that appear nowhere in the source.

The model is a fine-tuned ModernBERT-base (150M parameters). It runs on CPU in under a
second, which makes it practical as a post-generation check in production pipelines where
calling an LLM judge per answer is too slow or too expensive.

## Results

RAGTruth test set, n=2,500, example-level detection. F1 is for the hallucinated class;
the interval is a 2,000-resample bootstrap.

| Model | Params | F1 | Accuracy |
|---|---|---|---|
| GPT-4-turbo, zero-shot judge | API | 0.634 | — |
| GroundCheck (ModernBERT-base) | 150M | 0.682 (95% CI 0.66–0.71) | 0.747 |

Beyond RAGTruth, the model holds up on short inputs and minimal factual edits:
0.850 accuracy on the VitaminC test set (short claim-evidence pairs), and it catches
80% of grounded answers that had exactly one fact flipped (a number, date, direction
word, or named entity).

Trained on RAGTruth, VitaminC, and programmatically generated hard negatives. Weights:
[Pranshurs/groundcheck-modernbert](https://huggingface.co/Pranshurs/groundcheck-modernbert).

## Install

```bash
pip install groundcheck-rag
```

Without PyTorch installed, the library falls back to a lexical-overlap heuristic. Install
torch to use the model.

## Usage

As a library:

```python
from groundcheck import GroundCheck

gc = GroundCheck()  # downloads weights from Hugging Face on first use
result = gc.check(
    source="In Q3, revenue increased 12% year-over-year to $2.1 billion.",
    answer="Revenue fell 12% in Q3.",
)
print(result)  # hallucinated, with P(grounded)
```

From the command line:

```bash
groundcheck --source "..." --answer "..."
```

As a self-hosted service, for pipelines that want an HTTP endpoint instead of an
in-process library:

```bash
uvicorn groundcheck.api:app --port 8000
# POST /check  {"source": "...", "answer": "..."}
```

A single-page demo UI is served at the API root.

## How it works

The source (optionally prefixed with the user's question) and the answer are encoded as a
sequence pair; a binary classification head outputs P(grounded) vs P(hallucinated). The
model checks support against the provided source only — it is not a world-knowledge
fact-checker. An answer that is true but unsupported by the source is, by design,
hallucinated.

## Benchmark

`eval/benchmark.py` reproduces the RAGTruth numbers, including bootstrap confidence
intervals, latency, and cost per 1k checks.

## Limitations

- Tuned for recall: borderline answers tend to get flagged. If false alarms cost you
  more than misses, raise the decision threshold above 0.5.
- English only.
- Inputs past 512 tokens are truncated from the source side; very long documents should
  be checked chunk-wise.
- Research and non-commercial use. The training data (RAGTruth, VitaminC) carries
  research-oriented terms from its underlying sources.

## License

MIT. Base model: answerdotai/ModernBERT-base (Apache-2.0).
