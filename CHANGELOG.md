# Changelog

## 0.2.0 (unreleased)

Behaviour changes: read these before upgrading from 0.1.0.

- **No silent fallback.** The default backend is `model`. If torch/transformers are
  missing or the weights can't be loaded, `GroundCheck()` raises `ModelUnavailableError`
  and the CLI exits with status 2. 0.1.0 quietly answered with a lexical-overlap heuristic
  instead. The heuristic is still available, but only when requested
  (`backend="heuristic"`). `GROUNDCHECK_BACKEND=auto` is rejected.
- **Pinned weights.** The published model loads at Hub revision `998cec35`, so a later push
  to the Hub can't change results. The label map must be exactly
  `{0: grounded, 1: hallucinated}` or loading is refused. `model_info()` and
  `/api/health` report the requested and resolved revision.
- **Custom models.** A custom `model_path` no longer inherits the published model's
  revision, whether it's set via env, keyword or `DetectorSettings`.
- **Long answers.** An answer that can't fit the token budget raises `InputTooLongError`
  (422 from the API) instead of crashing the tokenizer. Only the source side is truncated.
- **API.** `uvicorn groundcheck.api:app` (documented in 0.1.0 but missing) now exists. The
  demo page ships in the wheel, the model loads at startup, and requests have input and
  batch limits.
- `GROUNDCHECK_MODEL_PATH=` (empty, as `.env` files often have it) now means the published
  model. Previously it switched to the heuristic.
- `max_length` stays at 2048 by default. 512 is the published evaluation protocol; see the
  README for both measurements.

Reproducibility and evidence (repository only, not in the wheel):

- `training/`: the v2 training recipe and deterministic, checksummed data builders over
  pinned public datasets.
- `MODEL_PROVENANCE.json` and `eval/provenance.py`: weight hashes, cross-checked against
  the Hub.
- `eval/reproduce.py`: reproduces the published numbers. Committed reports live in
  `eval/reports/`.
- `eval/gate.py`: a regression gate with committed reference and mutant evidence.
- `bench/latency.py`: measured CPU latency.
- CI: a unit tier on Python 3.9 and 3.12, plus a model tier (provenance, data, real-model
  tests, gate).
- Corrected documentation: the GPT-4 figure is an external reference from a different
  protocol, and latency is measured rather than claimed.

Licensing: the code is now Apache-2.0 (0.1.0 was MIT). The model weights stay MIT on the
Hub. Training and evaluation data keep their upstream terms; see `DATA_LICENSES.md`,
including the non-commercial terms on RAGTruth's MS MARCO and Yelp passages.

Verified stacks: torch 2.4.1 + transformers 4.49.0 (pinned; CI), and torch 2.14.1 +
transformers 5.18.0 (unpinned install, identical gate-slice predictions).

## 0.1.0

Initial release.
