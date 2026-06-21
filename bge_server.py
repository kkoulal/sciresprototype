"""
Local BGE-M3 embedding server for the ProductSimulation online RAG.

Exposes the exact /embed_both contract that online/retrieval.py expects:
    POST /embed_both  {"inputs": ["text", ...], "truncate": true}
        -> {"dense":  [[float, ...], ...],
            "sparse": [[{"index": int, "value": float}, ...], ...]}

CPU is fine for query-time embedding (one short query per request). The model
(~2.3 GB) downloads on first start; the first request is slow (model load),
then it's snappy.

Run:
    pip install FlagEmbedding fastapi "uvicorn[standard]"
    uvicorn bge_server:app --host 0.0.0.0 --port 8080
"""

import logging

from fastapi import FastAPI
from pydantic import BaseModel
from FlagEmbedding import BGEM3FlagModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("bge_server")

app = FastAPI(title="BGE-M3 embed server", version="1.0.0")

# Must be the SAME model used to embed the corpus, or query/doc vectors won't match.
# use_fp16=False -> CPU-safe. Loads on import (first /embed_both is slow otherwise).
log.info("Loading BAAI/bge-m3 (CPU)...")
_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
log.info("BGE-M3 ready.")


class EmbedReq(BaseModel):
    inputs: list[str]
    truncate: bool = True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/embed_both")
def embed_both(req: EmbedReq):
    out = _model.encode(
        req.inputs,
        return_dense=True,
        return_sparse=True,
        # query embedding is short; cap length when truncate=True for speed
        max_length=512 if req.truncate else 8192,
    )
    dense = [v.tolist() for v in out["dense_vecs"]]
    sparse = [
        [{"index": int(tok), "value": float(w)} for tok, w in lw.items()]
        for lw in out["lexical_weights"]
    ]
    return {"dense": dense, "sparse": sparse}
