# Retrieval corpora

**You do not need these to run Q-CARE.** The benchmark files under
`data/testbed/` already contain the top-30 retrieved chunks per query for both
`BM25` and `ANCE`, so the evaluation pipeline is fully self-contained.

The corpora below are only needed if you want to **re-run retrieval** from
scratch (e.g. to try a different retriever). Q-CARE uses eight source datasets,
four close-ended and four open-ended:

| Split | Dataset | Source |
|-------|---------|--------|
| close | NQ (Natural Questions) | https://github.com/beir-cellar/beir (`nq`) |
| close | HotpotQA | https://github.com/beir-cellar/beir (`hotpotqa`) |
| close | NewsQA | https://huggingface.co/datasets/lucadiliello/newsqa |
| close | FinQA | https://github.com/czyssrs/FinQA |
| open  | PubMedQA | https://github.com/pubmedqa/pubmedqa |
| open  | ELI5 | https://huggingface.co/datasets/eli5 |
| open  | LoTTE-science | https://github.com/stanford-futuredata/ColBERT (LoTTE) |
| open  | LoTTE-technology | https://github.com/stanford-futuredata/ColBERT (LoTTE) |

After downloading, place each corpus as `data/corpus/<DatasetName>/corpus.jsonl`
(BEIR format: one JSON object per line with `_id`, `title`, `text`). These
directories are git-ignored because of their size.
