# Is the regression gate real? Deliberate defects

Each mutant below was applied to `groundcheck/model.py` in a throwaway worktree of commit
`6c2c325`. The gate slice (`eval/baselines/gate_subset.json`, 200 RAGTruth + 100 VitaminC
rows) was scored with `eval.reproduce --subset`, and `eval.gate` was run against the
committed reference (`eval/baselines/v2-reference.json`). The code was restored afterwards;
none of these edits exist on the branch. The result JSONs record `"dirty": true` for the
mutated runs. Hardware: Apple M1, CPU, torch 2.4.1, transformers 4.49.0.

| Run | Defect (one-line diff) | Expected | Gate | Checks that failed |
|---|---|---|---|---|
| `clean` | none | pass | **PASSED** | agreement 1.0000, Δ score 0.00000 on both suites |
| `A_label_index` | score read from the hallucinated column: `self._grounded_index = 1` | every verdict inverted | **FAILED (12)** | agreement 0.00 on both suites; RAGTruth F1 0.66 → 0.25; VitaminC F1 0.88 → 0.10; both class recalls |
| `B2_drop_source` | premise built from the question only; the source is never sent | near-chance, collapse to "hallucinated" | **FAILED (10)** | agreement 0.40 / 0.52; recall_grounded 0.80 → 0.03 and 0.86 → 0.06. recall_hallucinated *rose*, which is why both class recalls are checked |
| `C_drop_question` | question omitted from the premise (a train/serve skew) | small, real shift | **FAILED (2)** | RAGTruth agreement 0.955 (< 0.98), mean Δ score 0.045 (> 0.01). F1 and accuracy stayed inside their floors; only the per-row checks catch this class. VitaminC has no questions and correctly passed |
| `D_other_revision` | same weights loaded from Hub revision `0c7dd063` instead of the pinned `998cec35` | provenance failure only | **FAILED (1)** | `provenance: revision`; every behavioural check passed, as it should, since the weights are byte-identical |
| `B_truncation` | token budget cut to `max_length // 4` | degraded scores | **not a gate catch**: the run crashed before scoring | Answers longer than 128 tokens no longer fit, and the tokenizer raised `Truncation error` (`B_truncation.crash.txt`). CI would still go red, but the gate itself never saw this mutant, which is why B2 replaced it |
