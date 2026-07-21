"""Q-CARE: query-agnostic, reference-free evaluation for retrieval-augmented generation.

Evaluates retrieval and generation under one principle - query coverage and
claim verifiability - by decomposing queries into sub-queries and answers into
atomic claims.

Typical use:

    from qcare import Backbone, QCAREPipeline

    pipeline = QCAREPipeline(Backbone().load())
    result = pipeline.evaluate(qid, query, answer, retrieved_chunks)
    print(result["metrics"])
"""

from .backbone import Backbone
from .metrics import compute_metrics
from .pipeline import QCAREPipeline
from .prompts import PromptLibrary

__all__ = ["Backbone", "QCAREPipeline", "PromptLibrary", "compute_metrics"]
__version__ = "1.0.0"
