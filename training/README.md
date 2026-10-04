# Training GroundCheck v2

This directory is the recipe behind the published weights
[`Pranshurs/groundcheck-modernbert`](https://huggingface.co/Pranshurs/groundcheck-modernbert)
(weights uploaded in Hub commit `0c7dd063`, card updated in `998cec35`). It is a cleaned-up
version of the Kaggle script that trained them. The changes are listed below, so you can
judge how close a rerun can get.

```bash
pip install -r requirements-train.txt
python -m training.build_data            # public data -> data/*.jsonl + data/manifest.v2.json
python -m training.build_data --verify   # rebuild and compare checksums with the committed manifest
python -m training.train                 # fine-tune (GPU strongly recommended)
python -m eval.reproduce --model-path artifacts/groundcheck-v2
```

## Inputs

| Input | Source | Pinned revision |
|---|---|---|
| Base model | `answerdotai/ModernBERT-base` (Apache-2.0) | `8949b909ec90` |
| RAGTruth | `wandb/RAGTruth-processed` | `eb4f4b9d1b68` |
| VitaminC | `tals/vitaminc` (CC BY-SA 3.0) | `be6febb761b0` |

Both datasets were last modified on the Hub in 2024 and 2022 respectively, before the v2 run
(2026-06-11), so these revisions are the data the published model saw. Nothing is
downloaded from anywhere else, and no LLM is used to generate data.

## The data mix (`configs/v2.json`)

| Part | Rows | How |
|---|---|---|
| RAGTruth train | 10,000 | Shuffled with seed 0. A response is `hallucinated` if any span is annotated. The question is dropped on ~50% of rows so `check(source, answer)` without a question matches training |
| VitaminC train | 16,000 | 10k SUPPORTS → `grounded`; 4k REFUTES + 2k NOT ENOUGH INFO → `hallucinated` |
| Hard negatives | 2,500 | One fact flipped in a grounded RAGTruth answer (number, direction word, date, or named entity). Rule-based, see `data.py` |
| **Train total** | **28,500** | Matches `train_n` in the published `metrics.json` |
| Validation | 2,000 | 1k RAGTruth + 1k VitaminC |

Test suites, used by `eval/`:

| Suite | Rows | Reproduced exactly? |
|---|---|---|
| `ragtruth_test` | 2,500 | **Yes.** It doesn't depend on any RNG; the first 2,500 usable rows of the test split |
| `vitaminc_test` | 2,000 | **Yes.** `test.shuffle(seed=0)`, first 2,000 usable rows |
| `flipped_test_*` | 500 + 500 | **No, it's regenerated.** See below |

## Hyperparameters

ModernBERT-base with a 2-class head, `{0: grounded, 1: hallucinated}`. The input is a pair
(`question + "\n\n" + source`, `answer`) with `truncation="only_first"` (the source is
truncated, never the answer) at `max_length=512`.

- 3 epochs, batch 16, learning rate 2e-5, linear schedule, warmup 6%, weight decay 0.01.
- Gradient checkpointing and `group_by_length`; fp16 on CUDA.
- Eager attention.
- Trainer seed 42. The original script left the seed at the Trainer default, which is 42.

The original run used a Kaggle P100 with torch 2.4.1 (cu121) and transformers 4.49.0, and
took about 54 minutes. Those versions are pinned in `requirements-train.txt`.

## Deviations from the original Kaggle script

1. **Deterministic hard negatives.** The original chose which fact to flip by iterating Python
   `set`s of strings. String-set order depends on `PYTHONHASHSEED`, which is randomised per
   process, so the original training hard negatives and flipped test pairs can't be
   regenerated. This was verified by running the unmodified algorithm under two hash seeds:
   different `train.jsonl` and `flipped_test_hallucinated.jsonl` hashes, but identical
   RAGTruth and VitaminC test hashes. `data.py` sorts those sets, so builds are now
   byte-identical across processes (`--verify` checks this). The flip *rules* are unchanged.
   Consequences:
   - The regenerated flipped holdout is a different sample from the one behind the published
     80.4% / 76.2%, so those two numbers can't be reproduced exactly.
   - The training set's hard negatives differ from the ones the published weights saw.
     Because a failed flip attempt consumes extra random draws, the VitaminC training
     sample drawn afterwards may differ too.
2. **Explicit pins**, for the base model, datasets and seeds, instead of whatever the Hub
   served on the day.
3. **Kaggle plumbing removed**: in-script `pip install`, `/kaggle/working` paths, zipping
   the output. On Kaggle, install `requirements-train.txt` with the cu121 torch wheel
   and run the same commands.
4. **Evaluation moved out** to `eval/reproduce.py`, so the published model and a
   retrained one are scored by the same code.

## What a rerun can and cannot show

GPU training is not bit-deterministic, and the hard negatives differ (deviation 1), so a
retrain gives a *sibling* of the published model, not the same weights. The published
weights are pinned by revision and file hash (`MODEL_PROVENANCE.json`) and evaluated
directly. The retrain path exists so the recipe can be inspected and rerun, not to claim
identical weights.
