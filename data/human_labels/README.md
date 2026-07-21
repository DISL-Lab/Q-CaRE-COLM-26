# Q-CARE Human Annotations

Human labels collected for the Q-CARE benchmark. Annotations were gathered on
Amazon Mechanical Turk over a 40-query-per-split sample of the testbed, using
`Qwen3-80B` decompositions as the items shown to annotators. No worker
identifiers or other personal information are included.

## Layout

```
human_labels/
├── mturk/                 # raw human annotations (ground truth)
│   └── <TargetModel>/
│       ├── [Query]test-<split>_annotation_Qwen3-80B_<TargetModel>_reasoning.csv
│       └── [Chunk]test-<split>_annotation_Qwen3-80B_<TargetModel>_reasoning.csv
└── scores/                # Q-CARE scores computed FROM the human labels
    ├── [<split>][<TargetModel>]query_level_results.json
    ├── [<split>]model_comparison_results.json
    └── metric_table_all.csv
```

`<TargetModel>` ∈ {GPT-5, Claude-Sonnet, Gemini-2.5-pro, GPT-oss_20B, Qwen3_30B,
Qwen3_4B, Gemma3_27B, Gemma3_4B}; `<split>` ∈ {close_ended, open_ended}.

## `mturk/` — raw annotations

Each CSV row batches ~10 annotation items plus one injected **attention check**.
List-valued columns are stored as stringified Python literals (parse with
`ast.literal_eval`); the i-th element of every list column corresponds to the
i-th `qid` in that row. Read with `encoding="utf-8-sig"` (files carry a BOM).

Columns:

| Column | Meaning |
|--------|---------|
| `idx`, `qid` | batch index and the list of query ids in the row |
| `query`, `decomposed_query`, `atomic_facts`, `retrieved_chunk`, `key_counts` | items shown to the annotator |
| `query_fact_relevance_check` | human label — is each atomic fact relevant to each sub-query |
| `chunk_fact_relevance_check` | human label — is each atomic fact supported by each retrieved chunk |
| `query_fact_coverage_check` | human label — do the atomic facts cover each sub-query |
| `query_chunk_coverage_check` | human label — do the retrieved chunks cover each sub-query |
| `*_check_reasoning` | free-text justification for each label |
| `attention_check` | marks the injected attention-check position in the batch |

The four `*_check` columns are the human labels for Q-CARE's four relevance
tasks. By construction the **[Query]** files carry the two query-side tasks
(`query_fact_relevance`, `query_fact_coverage`) and the **[Chunk]** files carry
the two chunk-side tasks (`chunk_fact_relevance`, `query_chunk_coverage`).

## `scores/` — human Q-CARE scores

Q-CARE metrics recomputed with human labels substituted for the model's
relevance judgments — i.e. the human-derived numbers reported in the paper.

- `[<split>]model_comparison_results.json`: per target model, `overall` and
  `by_dataset` aggregates of `completeness`, `conciseness`, `verifiableness`,
  `ndcg_at_10`.
- `[<split>][<TargetModel>]query_level_results.json`: per-query metric values
  (160 queries per split).
- `metric_table_all.csv`: flattened table across all models and splits.
