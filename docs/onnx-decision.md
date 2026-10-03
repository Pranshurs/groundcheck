# ONNX / int8: deferred, with reasons

**Decision (2026-10): don't ship an ONNX or quantised path in this release.**
Revisit it in the performance/MLOps phase, behind the same regression gate.

## Why not now

1. **This release is about provenance.** Every number in the README now comes from the
   pinned fp32 weights scored by `eval/reproduce.py`. A second inference path would need
   its own full evaluation (about 15 minutes per run on the reference machine) and its
   own gate reference, or the README would be reporting numbers for a model users aren't
   running.
2. **Quantisation changes predictions, and the gate is built to notice.** Dynamic int8
   on a classifier usually moves some scores across the 0.5 threshold. The gate allows
   only 2% prediction disagreement against the fp32 reference, so an int8 build must be
   gated against its *own* measured reference, after its accuracy delta on the full
   suites has been measured and accepted. That's a deliberate decision, not a side effect.
3. **Where the time goes is now measured** (`bench/results/`, Apple M1, CPU, 4 threads).
   Short claim-evidence pairs take p50 39 ms. Typical RAGTruth documents at the default
   2048-token budget take p50 349 ms / p95 825 ms. Documents over 2,048 tokens take
   ~1.45 s. At the 512-token published protocol everything long is ~210 ms. So the costly
   case is long documents, and it can already be traded off with `max_length` (an
   accuracy cost measured in `eval/reports/`) before introducing a second runtime.
   Whether int8 is worth its own accuracy cost should be decided on the target
   deployment hardware.
4. **Export risk is specific to this architecture.** ModernBERT uses unpadding and
   alternating local/global attention. The upstream `answerdotai/ModernBERT-base`
   publishes ONNX exports of the *base* model, which is evidence the architecture
   exports. But the fine-tuned head, the `only_first` truncation contract and numerical
   parity still need checking end to end.

## What doing it properly looks like

- `bench/onnx_export.py`: export the pinned revision with `optimum` and record opset,
  optimum, onnxruntime and model hashes.
- Parity first: the fp32 ONNX scores on the gate slice must agree ≥ 99.9% with PyTorch.
- Then int8 (dynamic, per-channel): run the full `eval.reproduce` suites and report the
  F1/accuracy delta with bootstrap CIs, plus latency with `bench.latency`, on the same
  machine.
- Ship int8 only as an opt-in backend (`backend="onnx-int8"`) with its own committed
  gate reference. The default stays the evaluated fp32 model.
