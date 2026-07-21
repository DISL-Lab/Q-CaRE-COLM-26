"""Interactive Q-CARE session.

Loads the backbone once so you can score queries in seconds instead of paying
the model-load cost per run. Useful for inspecting decompositions, atomic facts
and relevance judgements while adapting the prompts.

    CUDA_VISIBLE_DEVICES=0 python -i scripts/interactive.py

Then:

    ev("nq_test_812")                       # score one query
    r = ev("nq_test_812", "Claude-Sonnet")  # keep the full result
    r["decomposed_query"], r["atomic_facts"], r["relevance_check"]
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qcare import Backbone, PromptLibrary, QCAREPipeline  # noqa: E402
from qcare.backbone import DEFAULT_MODEL  # noqa: E402
from qcare.data import TARGET_MODELS, get_answer, get_chunks, get_gt_answer, load_testbed  # noqa: E402

MODEL = os.environ.get("QCARE_BACKBONE", DEFAULT_MODEL)

print(f"Loading backbone: {MODEL}")
_pipeline = QCAREPipeline(Backbone(MODEL).load(), PromptLibrary(), verbose=True)

_records = {}
for _split in ("close", "open"):
    for _r in load_testbed(os.path.join(ROOT, f"data/testbed/test-{_split}_ended_queries.json")):
        _records[_r["qid"]] = _r


def ev(qid, target_model="GPT-5", retrieval="BM25"):
    """Score one (query id, target model) pair and return the full result."""
    record = _records[qid]
    answer = get_answer(record, retrieval, target_model)
    if answer is None:
        raise KeyError(f"{qid} has no answer for {target_model}")
    result = _pipeline.evaluate(
        qid, record["query"], answer, get_chunks(record, retrieval), get_gt_answer(record)
    )
    return result


print(f"Ready. {len(_records)} queries loaded.")
print(f'Try: ev("nq_test_812")   targets: {", ".join(TARGET_MODELS)}')
