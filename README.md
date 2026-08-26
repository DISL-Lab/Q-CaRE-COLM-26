<div align="center">

<h1>Towards Query-Agnostic RAG Evaluation<br>via Query Coverage and Claim Verifiability</h1>

<h3><b>Q-CARE</b><br>: <ins>Q</ins>uery <ins>C</ins>overage and cl<ins>A</ins>im ve<ins>R</ins>ifiability for RAG <ins>E</ins>valuation</h3>

<b>Score a RAG system's retriever <i>and</i> generator in one pass.</b>

Jeonghwan Choi &nbsp;·&nbsp; Taewon Yun &nbsp;·&nbsp; Minjeong Ban &nbsp;·&nbsp; Gyeonghun Sun &nbsp;·&nbsp; Jae-Gil Lee &nbsp;·&nbsp; Hwanjun Song

Korea Advanced Institute of Science and Technology (KAIST)

[![Conference](https://img.shields.io/badge/COLM-2026-4b44ce.svg?style=flat-square)](https://colmweb.org/)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-Q--CARE-ffd21e.svg?style=flat-square)](https://huggingface.co/datasets/DISLab/Q-CARE)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![Backbone](https://img.shields.io/badge/🤗-Qwen3--30B--A3B-yellow.svg?style=flat-square)](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)

<br>
<img src="assets/overview.png" width="94%" alt="Q-CARE framework overview">

</div>

---

Q-CARE is a **query-agnostic**, fully **reference-free** framework for evaluating
retrieval-augmented generation. It decomposes queries into sub-queries and
answers into atomic claims, then scores retrieval and generation under one
principle — **query coverage** and **claim verifiability**. Because the scores
come from the query's own decomposition rather than a pre-annotated relevance
set, the same metrics apply from close-ended fact-seeking to open-ended
explanatory queries, and on to agentic requests that arrive in no fixed format —
and no gold answer is needed at evaluation time.

## 📥 What's released

| | Where | |
|---|---|---|
| **Evaluation pipeline** | `qcare/`, `evaluate.py` | one open-weight backbone, run locally — no API keys |
| **Prompts** | `configs/prompts.yaml` | every prompt, in one editable file |
| **Benchmark** | `data/testbed/` · [🤗 Hub](https://huggingface.co/datasets/DISLab/Q-CARE) | 800 queries with retrieved chunks and RAG answers, inline |

## 🔍 How it works

**Stage 1 · Decomposition.** A *decomposer* splits the query into minimal,
non-overlapping **sub-queries** and the answer into self-contained **atomic
claims**. Retrieval supplies the top-k chunks.

**Stage 2 · Coverage and verifiability.** An *alignment checker* makes four
judgements, one chunk and one sub-query at a time:

| Judgement | Question | Feeds |
|---|---|---|
| `query_chunk_coverage` | Does this chunk answer this sub-query? | C-Prec@k, C-nDCG@k |
| `query_fact_coverage` | Do the answer's claims cover this sub-query? | Completeness |
| `query_fact_relevance` | Which claims serve this sub-query? | Conciseness |
| `chunk_fact_relevance` | Does this chunk support this claim? | Verifiableness |

**Stage 3 · Metrics.** Coverage is aggregated per query. A chunk that covers
more sub-queries counts as *more* relevant — the graded relevance that makes the
retrieval metrics coverage-aware — while claim-level coverage and support give
the generator metrics.

Because everything is grounded in the query's own sub-questions, the same
procedure works whether the query has one factual answer or needs a long
explanation, and a gold answer is never consulted.

## 📊 Metrics

**Retriever.** Relevance is *graded*, not binary — a chunk is as relevant as the
share of the query it resolves:

<div align="center">
<b>r<sub>j</sub></b> &nbsp;=&nbsp; ( sub-queries covered by chunk <i>j</i> ) &nbsp;/&nbsp; ( total sub-queries )
</div>

<table>
<tr><td><b>C-Prec@k</b></td><td>mean of <i>r<sub>j</sub></i> over the top-k chunks</td></tr>
<tr><td><b>C-nDCG@k</b></td><td>nDCG over the same graded relevance, linear gain</td></tr>
</table>

A chunk that answers two of three sub-queries scores 2/3, outranking one that
answers a single sub-query — a distinction binary relevance cannot express.

**Generator.** Each metric is a ratio over the answer's own claims:

<table>
<tr><td><b>Completeness</b></td><td>sub-queries the answer covers &nbsp;/&nbsp; sub-queries</td></tr>
<tr><td><b>Conciseness</b></td><td>claims that serve a covered sub-query &nbsp;/&nbsp; claims</td></tr>
<tr><td><b>Verifiableness</b></td><td>claims supported by a retrieved chunk &nbsp;/&nbsp; claims</td></tr>
</table>

## ⚙️ Install

```bash
conda create -n qcare python=3.10
conda activate qcare

git clone https://github.com/DISL-Lab/Q-CaRE-COLM-26.git
cd Q-CaRE-COLM-26
pip install -r requirements.txt
```

**What you need.** Everything runs locally against one open-weight backbone — no
API keys, and no corpus download, since the retrieved chunks ship with the
benchmark. The default backbone `Qwen/Qwen3-30B-A3B-Instruct-2507` is pulled
from HuggingFace on first use and takes roughly 70 GB in bf16; point
`CUDA_VISIBLE_DEVICES` at several GPUs and it shards automatically.

`--eval_model` is the **backbone that makes the judgements** — the decomposer and
alignment checker — and is loaded locally with `transformers`, so it must be an
open-weight chat model on HuggingFace; API-only models cannot be passed here.
It is unrelated to `--target_model`, which names the **RAG system being scored**
(`GPT-5`, `Claude-Sonnet`, … — their answers ship with the benchmark, already
generated).

## 🚀 Quick start

One run evaluates **both halves of the RAG pipeline**: the same query
decomposition yields the retriever metrics for the chunks and the generator
metrics for the answer.

```bash
python evaluate.py \
  --input_path   data/testbed/test-close_ended_queries.json \
  --target_model GPT-5 \
  --output_dir   results
```

Each query yields the five metrics together with the intermediate judgements, so
a score can always be traced back to the sub-queries and claims behind it:

```jsonc
{
  "qid":          "PubMedQA_test_860",
  "query":        "is fetal anatomic assessment on follow-up antepartum sonograms clinically useful?",
  "model_answer": "Yes. In women with a prior normal anatomy scan, repeating a full anatomic survey on follow-up antepartum sonograms detected unanticipated anomalies in 7.1% overall and 12.3% when the follow-up was for fetal growth. In the growth group, 40% of detected anomalies led to neonatal interventions, indicating clinical usefulness.",

  // Stage 1 - the query splits in two; the answer breaks into four claims
  "decomposed_query": {
    "Core subquery1": "What is the clinical utility of fetal anatomic assessment during follow-up antepartum sonograms?",
    "Core subquery2": "What specific outcomes or benefits are associated with performing fetal anatomic assessment on follow-up antepartum sonograms?"
  },
  "atomic_facts": {
    "Atomic fact1": "Repeating a full anatomic survey on follow-up antepartum sonograms is clinically useful in women with a prior normal anatomy scan",
    "Atomic fact2": "Repeating a full anatomic survey on follow-up antepartum sonograms detected unanticipated anomalies in 7.1% of cases overall",
    "Atomic fact3": "Repeating a full anatomic survey on follow-up antepartum sonograms detected unanticipated anomalies in 12.3% of cases when the follow-up was for fetal growth",
    "Atomic fact4": "In the growth group, 40% of detected anomalies led to neonatal interventions"
  },

  // Stage 2 - alignment judgements (two of the four shown)
  "relevance_check": {
    "query_chunk_coverage": {
      "Chunk 1": ["Core subquery1"],                    // covers 1 of 2  ->  r = 0.5
      "Chunk 4": ["Core subquery1", "Core subquery2"]   // covers 2 of 2  ->  r = 1.0
    },
    "chunk_fact_relevance": {
      "Chunk 4": { "selected_facts": ["Atomic fact2", "Atomic fact3", "Atomic fact4"] }
    },
    "...": "..."
  },

  // Stage 3 - metrics
  "metrics": {
    // retriever: Chunk 4 resolves the whole query and Chunk 1 only half of it,
    // a distinction binary relevance cannot make
    "precision_at_10": 0.150,   // C-Prec@10
    "ndcg_at_10":      0.707,   // C-nDCG@10

    // generator
    "completeness":    1.000,   // both sub-queries answered
    "conciseness":     1.000,   // no claim is surplus to the query
    "verifiableness":  0.750    // 3 of 4 claims are supported by a chunk; fact 1 is
                                // the answer's own verdict, which no chunk states
  }
}
```

Full benchmark for one model, then a per-dataset table:

```bash
bash scripts/run_benchmark.sh GPT-5      # both splits, then the table
```

```
Model         Metric        NQ    NewsQA  HotpotQA   FinQA  Close Avg.
GPT-5         Comp.      0.755     0.687     0.758   0.163       0.591
GPT-5         Conc.      0.862     0.818     0.857   0.239       0.694
GPT-5         Veri.      0.600     0.787     0.690   0.599       0.669
```

<details>
<summary><b>Retriever-only mode</b></summary>

The run above already reports C-Prec@10 and C-nDCG@10 for the retriever it used.
When there is no generated answer to score — or the goal is to compare several
retrievers — this mode skips the answer-side steps, so it costs about half as
much and takes many retrievers at once. Conventional gold-chunk
Precision@10/nDCG@10 is computed alongside for reference.

```bash
python evaluate_retriever.py \
  --input_path  data/testbed/test-close_ended_queries.json \
  --retrievers  BM25,ANCE \
  --output_dir  results

python analysis/retriever_comparison.py --results_dir results
```

</details>

<details>
<summary><b>Interactive session</b></summary>

Loads the backbone once so each query scores in seconds — useful for inspecting
decompositions and judgements while adapting prompts.

```bash
python -i scripts/interactive.py
```

```python
>>> r = ev("nq_test_812")
>>> r["decomposed_query"]   # sub-queries
>>> r["atomic_facts"]       # claims
>>> r["relevance_check"]    # the alignment judgements
>>> r["metrics"]
```

</details>

## 🔧 Evaluate your own system

Nothing here is specific to the shipped benchmark — Q-CARE scores any
(query, answer, chunks) triple.

**Your generator**, against the same queries and chunks the eight shipped
systems saw:

```bash
python scripts/generate_answers.py \
  --input_path  data/testbed/test-close_ended_queries.json \
  --output_path data/testbed/test-close_ended_queries+mymodel.json \
  --model meta-llama/Llama-3.1-8B-Instruct --model_key MyModel

python evaluate.py \
  --input_path data/testbed/test-close_ended_queries+mymodel.json \
  --target_model MyModel
```

**Your retriever** — add its ranked chunks under a new key. Only the chunk texts
are needed, so no index or corpus has to leave your machine:

```python
record["retrieved_chunk"]["MyRetriever"] = my_search(record["query"], k=10)
```

```bash
python evaluate_retriever.py --retrievers MyRetriever,BM25 ...
```

**Your own queries**, straight from Python:

```python
from qcare import Backbone, QCAREPipeline

pipeline = QCAREPipeline(Backbone().load())
result = pipeline.evaluate(
    qid="example-1",
    query="Who wrote Dune and when was it published?",
    answer="Dune was written by Frank Herbert and published in 1965.",
    chunks=["Dune is a 1965 science-fiction novel by Frank Herbert...", "..."],
)
print(result["metrics"])
```

## 🎛️ Customising

| What | How |
|---|---|
| **Prompts** | Edit [`configs/prompts.yaml`](configs/prompts.yaml), or `--prompts my_prompts.yaml` |
| **Backbone** | `--eval_model <model id>` — any **instruction-tuned** chat model with a tokenizer chat template. The paper evaluates Qwen3-30B, Qwen3-80B, Llama3.1-8B and Llama3.3-70B as backbones |
| **Retriever** | `--retrieval_method ANCE` — BM25 and ANCE ship with the benchmark |
| **Parsing** | `--parsing paper` reproduces the published behaviour; `strict` never double counts a claim |

## 📦 Benchmark data

The benchmark ships in this repository under `data/testbed/` and is mirrored on
the Hugging Face Hub as
[**DISLab/Q-CARE**](https://huggingface.co/datasets/DISLab/Q-CARE), where the
same 800 records can be browsed in the dataset viewer or loaded directly:

```python
from datasets import load_dataset

close = load_dataset("DISLab/Q-CARE", "close_ended", split="test")
open_ = load_dataset("DISLab/Q-CARE", "open_ended", split="test")
```

The Hub copy flattens the nested keys for the viewer, so `retrieved_chunk` and
`model_prediction` become `retrieved_chunk_{bm25,ance}` and
`model_prediction_{bm25,ance}`. The pipeline below reads the original JSON
layout, which the Hub copy also carries verbatim under `raw/`.

### 800 queries, 8 datasets, 100 each

Half close-ended and half open-ended, so neither domain nor query style
dominates — and so the same metrics can be shown to hold across both.

| Close-ended · 4 × 100 | | Open-ended · 4 × 100 | |
|---|---|---|---|
| **NQ** | open-domain | **PubMedQA** | biomedical |
| **NewsQA** | news | **LoTTE-Science** | science forums |
| **HotpotQA** | multi-hop | **LoTTE-Technology** | technology forums |
| **FinQA** | finance, numerical | **ELI5** | long-form explanation |

### What each record holds

| Field | Contents |
|---|---|
| `query` | the question |
| `retrieved_chunk` | top-30 ranked chunks from **BM25** and **ANCE**, stored inline — **no corpus download needed** |
| `model_prediction` | answers from the eight RAG systems below, per retriever |
| `gt_answer`, `gt_chunk` | gold answer and gold chunks, carried for reference — Q-CARE never reads them |

| RAG systems | Pass to `--target_model` |
|---|---|
| Proprietary | `GPT-5` · `Claude-Sonnet` · `Gemini-2.5-pro` |
| Open-weight | `GPT-oss_20B` · `Qwen3_30B` · `Qwen3_4B` · `Gemma3_27B` · `Gemma3_4B` |

## 🗂️ Repository layout

```
evaluate.py                 score a target model (retriever + generator metrics)
evaluate_retriever.py       retriever-only evaluation
configs/prompts.yaml        every prompt the pipeline uses
qcare/
  backbone.py               model loading and generation
  prompts.py                prompt loading and rendering
  pipeline.py               decomposition, alignment checks, scoring
  parsing.py                response parsers
  metrics.py                metric definitions
  data.py                   benchmark access helpers
analysis/                   per-dataset table, retriever ranking
scripts/                    benchmark runner, answer generation, interactive session
docs/REPRODUCE.md           reproducing the paper's tables
```

## 📄 Citation

```bibtex
@inproceedings{choi2026qcare,
  title     = {Towards Query-Agnostic {RAG} Evaluation via Query Coverage and Claim Verifiability},
  author    = {Choi, Jeonghwan and Yun, Taewon and Ban, Minjeong and
               Sun, Gyeonghun and Lee, Jae-Gil and Song, Hwanjun},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2026}
}
```

## ⚖️ License

Released under the MIT License — see [LICENSE](LICENSE).
