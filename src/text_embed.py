"""Sentence-transformer embedding of each transcript (semantic/syntactic content).

Writes features/text_emb.npz with keys + X [N, 768].
"""
import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "sentence-transformers/all-mpnet-base-v2"
rows = [json.loads(l) for l in open("features/transcripts.jsonl")]
m = SentenceTransformer(MODEL)
X = m.encode([r["text"] for r in rows], batch_size=16, show_progress_bar=True, normalize_embeddings=True)
np.savez("features/text_emb.npz", keys=np.array([r["key"] for r in rows]), X=X.astype(np.float32))
print("saved", X.shape)
