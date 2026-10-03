# GroundCheck

[![tests](https://github.com/Pranshurs/groundcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/Pranshurs/groundcheck/actions/workflows/ci.yml)
[![model](https://github.com/Pranshurs/groundcheck/actions/workflows/model.yml/badge.svg)](https://github.com/Pranshurs/groundcheck/actions/workflows/model.yml)
[![PyPI](https://img.shields.io/pypi/v/groundcheck-rag)](https://pypi.org/project/groundcheck-rag/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Checks whether a RAG answer is supported by the source it was given, using a fine-tuned
150M-parameter ModernBERT classifier that runs on CPU.

## What it does

Given a source text and an answer (and optionally the user's question), GroundCheck
returns `grounded` or `hallucinated` with P(grounded). It checks support against the
provided source only, so a true statement the source doesn't contain counts as
`hallucinated`. It is aimed at the common RAG failures: a changed number, a flipped
direction ("increased" → "decreased"), a wrong date or entity, a claim that appears nowhere.

## Model

- ModernBERT-base (`answerdotai/ModernBERT-base`) with a 2-class head, `{0: grounded, 1: hallucinated}`.
- Weights: [`Pranshurs/groundcheck-modernbert`](https://huggingface.co/Pranshurs/groundcheck-modernbert),
  pinned in the package to revision `998cec35` (weights committed in `0c7dd063`).
- File hashes, architecture and the run's own training arguments: [`MODEL_PROVENANCE.json`](MODEL_PROVENANCE.json).
- Recipe and data: [`training/`](training/README.md). Model card: [`MODEL_CARD.md`](MODEL_CARD.md).

## Results

### Measured in this repository

Pinned weights, test sets rebuilt from pinned public datasets (`data/manifest.v2.json`),
CPU (Apple M1), torch 2.4.1 / transformers 4.49.0. F1 is for the `hallucinated` class;
the 95% CIs come from a 2,000-resample bootstrap.

| Suite | n | max_length 512 (published protocol) | max_length 2048 (package default) |
|---|---|---|---|
| RAGTruth test (first 2,500 of 2,700) | 2,500 | F1 **0.682** [0.658, 0.705], acc 0.746 | F1 **0.696** [0.672, 0.718], acc 0.758 |
| VitaminC test | 2,000 | acc 0.850, F1 0.845 [0.828, 0.862] | identical (all inputs are short) |
| One-fact flips caught¹ | 500 | 78.0% | 87.2% |
| Same answers unflipped, kept grounded¹ | 500 | 76.0% | 74.0% |
| Hand-written manual cases | 5 | 5/5 | — |

Sources: [`eval/reports/v2-reproduction-512.json`](eval/reports/v2-reproduction-512.json)
and [`eval/reports/v2-maxlength-2048.json`](eval/reports/v2-maxlength-2048.json). Each
records the code commit, model revision, weight hash, data hashes and environment.

At 512 these match the numbers the training run published (RAGTruth F1 0.6824, acc 0.7468;
VitaminC acc 0.8495) to within one prediction out of 2,500. 2048 versus 512 on the same rows
is a paired RAGTruth ΔF1 of +0.014 (95% CI −0.001 to +0.028), at about 2.5× the CPU time
on long documents.

¹ A regenerated holdout. The original training run sampled its flipped pairs in an order
that depended on Python's per-process hash seed, so that exact sample can't be rebuilt; it
reported 80.4% / 76.2%. See [`training/README.md`](training/README.md#deviations-from-the-original-kaggle-script).

### External published reference (not a controlled comparison)

| System | RAGTruth F1 | Source |
|---|---|---|
| GPT-4-turbo, zero-shot prompt judge | 0.634 (P 0.469, R 0.979) | RAGTruth paper (Niu et al., 2024), as tabulated in LettuceDetect (2025) |

This figure comes from a different protocol: a prompted LLM judge scored on all 2,700
RAGTruth test responses. GroundCheck's numbers above use the first 2,500 responses (as the
training run did). The two weren't run head-to-head, so the comparison is for orientation
only and doesn't support a "beats GPT-4" claim. `eval/benchmark.py`
can run a protocol-matched LLM judge on the same rows if you supply an API key.

## Quick start

```bash
pip install "groundcheck-rag[model]"
```

```python
from groundcheck import GroundCheck

gc = GroundCheck()  # downloads the pinned weights on first use (~600 MB)
r = gc.check(
    source="In Q3, revenue increased 12% year-over-year to $2.1 billion.",
    answer="Revenue fell 12% in Q3.",
)
print(r.label, r.grounded_score, r.backend)  # hallucinated <score> model
```

If torch/transformers aren't installed or the weights can't be loaded, `GroundCheck()`
raises `ModelUnavailableError`. It never silently falls back. A dependency-free
lexical-overlap baseline is available, but only if you ask for it:
`GroundCheck(backend="heuristic")`. Every result records which backend produced it.

Answers longer than the token budget raise `InputTooLongError`. Only the source side is
ever truncated, so split long answers into separate claims.

## API

```bash
pip install "groundcheck-rag[model,serve]"
uvicorn groundcheck.api:app --port 8000
```

- `POST /api/check`: `{"source": "...", "answer": "...", "question": "..."}` → result
- `POST /api/check_batch`: `{"items": [...]}`, up to 64 items
- `GET /api/health`: backend, pinned and resolved revision, max_length

A demo page is served at `/`. Invalid or oversized input returns 422. If the model can't be
loaded, the server doesn't start. A `Dockerfile` installs the package with `[serve,model]`
and runs the same command; it hasn't been built as part of this release's verification.

## CLI

```bash
groundcheck --source "Paris is the capital of France." --answer "France's capital is Paris." --json
groundcheck --backend heuristic --source-file ctx.txt --answer-file ans.txt
```

The CLI exits with status 2 if the model is unavailable.

## Configuration

| Variable | Default | |
|---|---|---|
| `GROUNDCHECK_BACKEND` | `model` | `model` or `heuristic`; nothing else is accepted |
| `GROUNDCHECK_MODEL_PATH` | published model | HF id or local directory |
| `GROUNDCHECK_MODEL_REVISION` | `998cec35…` for the published model | HF revision |
| `GROUNDCHECK_MAX_LENGTH` | `2048` | 512 = published protocol, faster on long documents |
| `GROUNDCHECK_THRESHOLD` | `0.5` | grounded if P(grounded) ≥ threshold |

## Reproduce everything

```bash
pip install -r requirements-train.txt -r requirements-dev.txt && pip install -e ".[serve]"
python -m eval.provenance --verify          # weights match MODEL_PROVENANCE.json and the Hub's LFS hash
python -m training.build_data --verify      # rebuild the public data; checksums must match the manifest
pytest -m model                             # real-model tests
python -m eval.reproduce                    # full evaluation at the 512-token published protocol (~15 min on an M1 CPU)
python -m eval.reproduce --max-length 2048  # the package default
python -m training.train                    # retrain (GPU strongly recommended; gives a sibling model, see training/README.md)
```

## Regression gate

`eval.gate` compares a run on a fixed, stratified slice (200 RAGTruth + 100 VitaminC rows)
with a committed reference run. It checks provenance, per-row prediction agreement (≥ 98%),
mean score drift (≤ 0.01), F1/accuracy floors and per-class recall floors; the reasoning
for each threshold is in the module docstring. CI runs it on every change that can affect
predictions.

It has been shown to fail on deliberate defects: an inverted label index, a dropped source,
a dropped question (caught only by the per-row checks), and the right weights loaded from
an unpinned revision. Details: [`eval/reports/gate-mutants/`](eval/reports/gate-mutants/README.md).

## Performance

Measured on one machine (Apple M1, 8 cores, 16 GB, CPU, 4 torch threads, not otherwise
idle), single requests after warm-up, with real test rows grouped by pair length
([`bench/results/`](bench/results/)):

| Pair length | max_length 2048: p50 / p95 | max_length 512: p50 / p95 |
|---|---|---|
| ≤ 128 tokens | 39 / 52 ms | 38 / 49 ms |
| 129–512 | 172 / 208 ms | 172 / 206 ms |
| 513–2,048 | 349 / 825 ms | 210 / 212 ms |
| > 2,048 (n=14) | 1,451 / 1,460 ms | 212 / 214 ms |

Loading the model from the local cache took 1.1–1.5 s. Run `python -m bench.latency` on
your own hardware; other CPUs will differ. There's no ONNX or quantised path yet; see
[`docs/onnx-decision.md`](docs/onnx-decision.md).

## Architecture

```
groundcheck/    library: model + heuristic backends, CLI, FastAPI app
training/       data builders (pinned public datasets) and the fine-tuning recipe
eval/           reproduction harness, provenance check, regression gate, LLM-judge benchmark
bench/          CPU latency harness
tests/          unit tier (no weights) and model tier (pytest -m model)
```

## Limitations

- **Truncation.** 76% of RAGTruth test pairs exceed 512 tokens and 19 of 2,500 exceed 2,048.
  Beyond the budget the source is cut from the end, so evidence late in a long document is
  never seen. Chunk long documents.
- **Calibration.** Precision on RAGTruth is about 0.63–0.64, so roughly a third of
  `hallucinated` verdicts on long RAG answers are false alarms. Scores aren't calibrated
  probabilities; tune the threshold on your own data.
- **Trade-off on minimal edits.** About a quarter of unedited grounded answers in the
  flipped-pair holdout are still flagged.
- **Scope.** English only. Answer-level verdicts, not span-level. It checks support from the
  source, not truth.
- **Benchmark coverage.** RAGTruth, VitaminC and rule-generated single-fact flips. No
  adversarial or out-of-domain evaluation has been run.
- **Licensing.** The training data (RAGTruth, VitaminC) carries research-oriented terms from
  its underlying sources. Research and non-commercial use.

## Development

```bash
pip install -r requirements-dev.txt && pip install -e .
pytest                 # unit tier: no torch, no weights
pytest -m model        # model tier: needs requirements-train.txt, downloads the weights
ruff check --select E,F,W,B --ignore E501 groundcheck eval training bench tests run.py
```

## Hosted version

A commercially licensed version (sentence-level verdicts, batch log auditing, hosted API)
is in pilot. Details are in docs/how-it-was-built.md, or email pranshu.rs08@gmail.com.

## License

MIT. Base model: answerdotai/ModernBERT-base (Apache-2.0).
