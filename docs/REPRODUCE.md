# Reproducing the paper

Two levels of reproduction are available.

**Scoring reproduction** — re-derive the metrics from the relevance judgements
released with the benchmark. This is exact: the numbers match the paper to the
last digit, and it needs no GPU.

**Full reproduction** — re-run the backbone over the benchmark so it regenerates
every decomposition and judgement, then score. Because the metrics are produced
by an LLM, a fresh run is close to, but not bit-identical with, the published
numbers.

---

## Environment

```bash
pip install -r requirements.txt
```

The pinned versions are the ones used for the published results:
`torch 2.11.0`, `transformers 5.7.0`, `accelerate 1.13.0`.

Backbone: `Qwen/Qwen3-30B-A3B-Instruct-2507`, greedy decoding
(`temperature=0.0`), `max_new_tokens=2000`, seed 1234. Given the same backbone,
library versions and hardware, a run is deterministic.

---

## Full reproduction: generator benchmark

Evaluate one target model over all eight datasets (800 queries) and aggregate:

```bash
bash scripts/run_benchmark.sh GPT-5
```

The script evaluates both splits and writes the aggregated table. Runs
checkpoint every five queries and resume automatically, so it is safe to
interrupt. The two splits are independent, so they can equally well be launched
side by side on separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py --input_path data/testbed/test-close_ended_queries.json --target_model GPT-5 &
CUDA_VISIBLE_DEVICES=1 python evaluate.py --input_path data/testbed/test-open_ended_queries.json  --target_model GPT-5 &
wait
```

For the complete table, repeat for each target model:

```
Qwen3_4B  Qwen3_30B  Gemma3_4B  Gemma3_27B
GPT-oss_20B  GPT-5  Gemini-2.5-pro  Claude-Sonnet
```

then aggregate everything at once:

```bash
python analysis/generator_benchmark.py --results_dir results --out results/generator_benchmark.csv
```

---

## Retriever evaluation

The retriever metrics come from the query decomposition and the retrieved
chunks, never from the answer — so the generator run above already reports
C-Prec@10 and C-nDCG@10 for whichever retriever it used, and those values are
identical no matter whose answer is being scored.

To compare retrievers, `evaluate_retriever.py` runs only the two steps the
retrieval metrics need, which costs about half as much and takes several
retrievers in one pass. Conventional binary Precision@10/nDCG@10 against the
gold chunks is computed alongside, from the `gt_chunk` field in the testbed, so
both rankings come out of the same run.

```bash
# score the retrievers shipped with the benchmark, on both splits
python evaluate_retriever.py \
  --input_path  data/testbed/test-close_ended_queries.json \
  --retrievers  BM25,ANCE \
  --output_dir  results

python evaluate_retriever.py \
  --input_path  data/testbed/test-open_ended_queries.json \
  --retrievers  BM25,ANCE \
  --output_dir  results

# rank them and compare the two rankings
python analysis/retriever_comparison.py --results_dir results --out results/retriever_comparison.csv
```

The comparison prints, per retriever, the mean coverage-aware and conventional
scores, the rank each metric assigns, and the shift between them, plus the
Spearman correlation between the two rankings.

The paper evaluates a wider set of retrievers than the two whose rankings ship
with the benchmark. To reproduce those rows, add each retriever's ranked chunks
to the testbed under its own key and pass that key to `--retrievers`:

```python
record["retrieved_chunk"]["SPLADE"] = [chunk1, chunk2, ...]   # top-10, ranked
```

The same route evaluates **your own retriever** — no corpus or index needs to be
shared, only the ranked chunk texts.

---

## Human agreement

Correlates Q-CARE's judgements with the human annotations in
`data/human_labels/`, for each backbone and each metric:

```bash
python analysis/human_agreement.py \
  --relevance graded \
  --human_dir   <human_labelled_dir> \
  --llm_root    <backbone_judgement_root> \
  --backbones   Qwen3-30B,Llama3.1-8B,Llama3.3-70B,Qwen3-80B \
  --sample_qids data/human_labels/annotated_qids.json \
  --corr pearson
```

`--relevance graded` uses the coverage-aware relevance the paper reports;
`--relevance binary` reproduces the binary ablation. `--sample_qids` restricts
the comparison to the 320 human-annotated queries and is required for a faithful
result.

---

## Notes on exactness

* **Greedy decoding still varies slightly** across GPU models and library
  versions. Expect per-query differences on a minority of queries; dataset-level
  averages are stable.
* **`--parsing paper` (default)** reproduces the published behaviour, including
  the case where a fact is named once by key and once by text. `--parsing
  strict` resolves fact text back to keys so nothing is double counted; it is
  cleaner but yields higher Verifiableness than the published numbers.
* Metrics never use the gold answer; `gt_answer` is carried through the results
  for reference only.
