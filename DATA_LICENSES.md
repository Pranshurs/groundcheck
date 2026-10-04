# Data sources and their terms

None of these datasets is redistributed in this repository. `training/build_data.py`
downloads them from the Hugging Face Hub at the pinned revisions in `data/manifest.v2.json`.
The files it writes under `data/` keep the upstream terms below. They are **not** covered
by this repository's Apache-2.0 license, and the model weights' MIT license doesn't
relicense them.

| Source | Pinned at | Terms | Used for |
|---|---|---|---|
| RAGTruth: `wandb/RAGTruth-processed` (upstream [ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth)) | `eb4f4b9d` | MIT for the dataset and annotations. See the rows below for the underlying passages. | training (10,000 rows), RAGTruth test (2,500), flipped pairs |
| … passages from **MS MARCO** | — | Microsoft: "non-commercial research purposes only" | the QA portion of RAGTruth |
| … passages from the **Yelp Open Dataset** | — | Yelp Dataset Terms: academic / non-commercial use | the data-to-text portion |
| … articles from **CNN/DailyMail** | — | Copyright of the publishers; distributed for research | the summarisation portion |
| … responses from GPT-3.5/GPT-4, Llama-2-chat and Mistral-7B | — | The respective model providers' terms. The Llama 2 licence restricts using outputs to improve other large language models. | answer texts |
| VitaminC: `tals/vitaminc` | `be6febb7` | CC BY-SA 3.0 (Wikipedia-derived). Attribution and share-alike apply if you redistribute it. | training (16,000 rows), VitaminC test (2,000) |
| `eval/cases/sample_grounding.jsonl` | in this repo | Written for this repository; Apache-2.0 | manual examples |

**Citations:**
- Niu et al., 2024, *RAGTruth: A Hallucination Corpus for Developing Trustworthy
  Retrieval-Augmented Language Models*.
- Schuster et al., NAACL 2021, *Get Your Vitamin C! Robust Fact Verification with
  Contrastive Evidence*.

This summary is engineering diligence, not legal advice.
