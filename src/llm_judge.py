"""LLM-as-a-judge + LLM embeddings from a single forward pass per transcript.

A local instruction-tuned LLM (MLX, Apple Silicon) reads SHL's grammar rubric and the
transcript. From that one forward pass we keep:
1. Judge score: next-token distribution over the digits "1".."5" -> expected score
   E[s] = sum_s s * p(s) and the 5 probabilities. Using the probability-weighted
   expectation (not a sampled answer) gives a continuous, deterministic score.
   Zero-shot: no training labels are used.
2. Embeddings: hidden states mean-pooled over the transcript tokens at 4 depths
   (1/4, 1/2, 3/4, last layer). The LLM has read the rubric before the transcript,
   so these states are "grammar-aware"; a ridge on top is trained later with CV.

Usage: python src/llm_judge.py [mlx model id]
Writes features/llm_<model>.csv (judge) and features/llm_<model>_emb.npz (embeddings)
"""
import json, os, sys
import numpy as np, pandas as pd
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask

MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

RUBRIC = """Grammar Score rubric:
1 - The person's speech struggles with proper sentence structure and syntax, displaying limited control over simple grammatical structures and memorized sentence patterns.
2 - The person has a limited understanding of sentence structure and syntax. Although they use simple structures, they consistently make basic sentence structure and grammatical mistakes. They might leave sentences incomplete.
3 - The person demonstrates a decent grasp of sentence structure but makes errors in grammatical structure, or they show a decent grasp of grammatical structure but make errors in sentence syntax and structure.
4 - The person displays a strong understanding of sentence structure and syntax. They consistently show good control of grammar. While occasional errors may occur, they are generally minor and do not lead to misunderstandings; the person can correct most of them.
5 - Overall, the person showcases high grammatical accuracy and adept control of complex grammar. They use grammar accurately and effectively, seldom making noticeable mistakes. Additionally, they handle complex language structures well and correct themselves when necessary."""

SYSTEM = "You are an expert English-language assessor who rates the grammar of spoken responses."


def prompt(text):
    return (f"{RUBRIC}\n\nBelow is an automatic transcript of a person speaking for about one minute. "
            f"Ignore punctuation and capitalisation (they were added by the speech recogniser) and judge only "
            f"the grammar of what the person said.\n\nTranscript:\n\"\"\"{text}\"\"\"\n\n"
            f"Answer with a single digit from 1 to 5: the grammar score.")


mx.set_cache_limit(1 << 30)  # cap MLX's buffer cache (1 GB): unbounded it grows to >10 GB on a 16 GB Mac
_M = {}


def _load(name=MODEL):
    if _M.get("name") != name:
        model, tok = load(name)
        L = len(model.model.layers)
        _M.update(name=name, model=model, tok=tok,
                  digit_ids=mx.array([tok.encode(str(d), add_special_tokens=False)[-1] for d in range(1, 6)]),
                  KEEP=[L // 4, L // 2, 3 * L // 4, L])  # hidden-state depths to pool (L = final, after norm)
    return _M


def forward(text, name=MODEL):
    """Rubric-prompted forward pass: (log-probs of digits 1-5, [4, H] pooled transcript hidden states)."""
    M = _load(name); model, tok, KEEP = M["model"], M["tok"], M["KEEP"]
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(text)}]
    s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    a = s.index('"""') + 3; b = s.index('"""', a)  # transcript character span
    t0 = len(tok.encode(s[:a], add_special_tokens=False))
    t1 = max(len(tok.encode(s[:b], add_special_tokens=False)), t0 + 1)
    ids = mx.array(tok.encode(s, add_special_tokens=False))[None]

    h = model.model.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    pooled = []
    for i, layer in enumerate(model.model.layers, 1):
        h = layer(h, mask, None)
        if i in KEEP[:-1]:
            pooled.append(h[0, t0:t1].mean(0))
    h = model.model.norm(h)
    pooled.append(h[0, t0:t1].mean(0))
    last = h[:, -1:]
    logits = (model.lm_head(last) if hasattr(model, "lm_head") else model.model.embed_tokens.as_linear(last))[0, -1]
    lp = logits[M["digit_ids"]].astype(mx.float32)
    emb = mx.stack(pooled).astype(mx.float32)
    mx.eval(lp, emb)
    return np.array(lp), np.array(emb)


def judge_table(LP):
    """Log-probs [N, 5] -> DataFrame of llm_expected and llm_p1..llm_p5."""
    LP = np.atleast_2d(LP); p = np.exp(LP - LP.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    df = pd.DataFrame(p, columns=[f"llm_p{d}" for d in range(1, 6)])
    df.insert(0, "llm_expected", p @ np.arange(1, 6))
    return df


if __name__ == "__main__":
    MODEL = sys.argv[1] if len(sys.argv) > 1 else MODEL
    NAME = MODEL.split("/")[-1]
    CKPT = f"features/llm_{NAME}.partial.npz"
    rows = [json.loads(l) for l in open("features/transcripts.jsonl")][:int(os.environ.get("LIMIT", 10**9))]  # LIMIT: smoke test
    keys, LP, E = [], [], []
    if os.path.exists(CKPT):
        z = np.load(CKPT); keys, LP, E = list(z["keys"]), list(z["lp"]), list(z["emb"])
    done = set(keys)
    for r in rows:
        if r["key"] in done:
            continue
        lp, emb = forward(r["text"] or "(no speech)", MODEL)
        keys.append(r["key"]); LP.append(lp); E.append(emb.astype(np.float16))
        if len(keys) % 50 == 0:
            np.savez(CKPT, keys=np.array(keys), lp=np.stack(LP), emb=np.stack(E))
            print(len(keys), flush=True)

    df = judge_table(np.stack(LP)); df.insert(0, "key", keys)
    df.to_csv(f"features/llm_{NAME}.csv", index=False)
    np.savez(f"features/llm_{NAME}_emb.npz", keys=np.array(keys), X=np.stack(E), layers=np.array(_load(MODEL)["KEEP"]))
    os.path.exists(CKPT) and os.remove(CKPT)
    print("saved", df.shape, np.stack(E).shape)
