# A 150M model that beats GPT-4-as-judge at catching RAG hallucinations — trained for $0

I built GroundCheck, a small open model that checks whether an AI answer is actually
supported by the source it cites. It scores 0.682 F1 on the RAGTruth benchmark, ahead of
the published GPT-4-turbo-as-judge baseline (0.634), and it returns a verdict in under a
second on a laptop CPU. Total compute cost: zero — every training run fit inside Kaggle's
free GPU quota.

Weights, benchmark code, and a pip package are public. This post is the honest version of
how it went, including the part where the first model failed.

## The problem

RAG pipelines answer questions from documents, and sometimes they state things the
documents never said: a number quietly changed, "increased" turned into "decreased," a
plausible fact invented from nowhere. Checking every answer with a frontier LLM-as-judge
works, but it is slow (seconds), priced per token, and ships your data to a third party.

This is a narrow classification task, and narrow tasks are where small specialized models
earn their keep. Premise: source document (plus the user's question when available).
Hypothesis: the answer. Output: grounded or hallucinated, with a probability.

## v1: good benchmark, bad model

The first version was ModernBERT-base fine-tuned on RAGTruth, the standard benchmark for
this task: 0.688 F1, clear of the GPT-4 judge. Shipped it, felt great.

Then I ran five quick manual cases — short, realistic inputs like a source saying revenue
"increased 12%" and an answer claiming it "fell 12%", or a vaccine's "94% effective"
becoming "49% effective". The model called all five grounded. Two out of five correct,
and the misses were the embarrassing kind.

The diagnosis was ordinary distribution shift. RAGTruth is long RAG documents, so the model
had learned that distribution and nothing else. On short inputs it defaulted to "grounded."
It also trained with a question always prepended, while the library's main entry point
sends none — a silent train/inference mismatch on every real call.

## v2: fix the data, not the model

The fix was three changes to the training mix, none to the architecture:

1. **Short claim–evidence pairs** (VitaminC): sentences that differ by one small factual
   edit — exactly the failure mode from the manual test.
2. **Programmatic hard negatives**: take a grounded answer, flip exactly one fact — a
   number, a date, a direction word, a named entity — and label it hallucinated. The
   original stays in the training set, so the model must compare rather than
   pattern-match. ~2,700 of these flips are number edits, which is precisely where v1
   was blind.
3. **Question dropout**: half the training rows lose their question, so no-question
   inference matches training.

One retraining run later (28.5k examples, 3 epochs, a single free P100, under two hours):
RAGTruth held at 0.682 F1 — statistically tied with v1 — while VitaminC accuracy hit 0.850,
80% of single-fact flips get caught, and the manual five went 5/5 with confident margins.
The first rebalance attempt actually overshot (it flagged too many correct answers), which
cost one more run to fix; the intermediate numbers are in the repo history.

The general lesson, which everyone repeats and I now believe: with a fixed small model,
the data mix is the steering wheel. Three data changes moved real-world behavior more
than any architecture choice would have.

## What it doesn't do

Honest limitations, so you can decide if it's useful before installing:

- It verifies support against the provided source only. A true statement not in the source
  is, by design, hallucinated.
- It trades some precision for recall on long fact-dense documents — it flags more
  borderline answers than v1 did. If false alarms cost you more than misses, raise the
  threshold.
- English only, 512-token source window (chunk longer documents).
- The training data carries research-oriented licenses, so the open model is for research
  and non-commercial use.

## Try it

```bash
pip install groundcheck-rag
```

```python
from groundcheck import GroundCheck
gc = GroundCheck()
gc.check(
    source="In Q3, revenue increased 12% year-over-year to $2.1 billion.",
    answer="The company's revenue fell 12% in Q3.",
)   # -> hallucinated
```

- Code and benchmark harness: https://github.com/Pranshurs/groundcheck
- Weights: https://huggingface.co/Pranshurs/groundcheck-modernbert

The benchmark script reproduces every number above, confidence intervals included.

I'm also piloting a hosted version trained from scratch on commercially-clean data
(sentence-level verdicts, batch log auditing, commercial use allowed). If that's relevant
to your team, email me — address is in the repo.
