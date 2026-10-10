"""Interpretable features from Whisper transcripts.

1. LanguageTool grammar-error counts (normalised per 100 words), by category.
2. Fluency / complexity: words, words per second, pause stats, fillers,
   mean sentence length, type-token ratio, long-word ratio.
3. ASR confidence: Whisper avg logprob and word probabilities
   (unclear / disfluent speech -> lower confidence).
4. GPT-2 perplexity of the transcript (ungrammatical text is less probable).
Writes features/text_feats.csv. Incremental: clips already in the CSV are skipped, so it
can run on the transcripts available so far and be topped up later.
"""
import glob, json, os, re
from collections import Counter
import numpy as np, pandas as pd, torch
import language_tool_python
from transformers import AutoTokenizer, AutoModelForCausalLM

OUT = "features/text_feats.csv"
IGNORE = {"UPPERCASE_SENTENCE_START", "WHITESPACE_RULE", "COMMA_PARENTHESIS_WHITESPACE",
          "EN_QUOTES", "PUNCTUATION_PARAGRAPH_END", "MORFOLOGIK_RULE_EN_US"}
CATS = ["GRAMMAR", "TYPOS", "PUNCTUATION", "STYLE", "CONFUSED_WORDS", "REDUNDANCY", "COLLOCATIONS", "MISC"]
FILLERS = {"um", "uh", "erm", "er", "ah", "hmm", "mm", "uhm"}
dev = "cpu"  # GPT-2 small is fast on CPU; keeps the GPU free for the big models
_M = {}


def _models():
    if not _M:
        _M["tool"] = language_tool_python.LanguageTool("en-US")
        _M["tok"] = AutoTokenizer.from_pretrained("gpt2")
        _M["lm"] = AutoModelForCausalLM.from_pretrained("gpt2").to(dev).eval()
    return _M


def perplexity(text):
    m = _models()
    ids = m["tok"](text, return_tensors="pt").input_ids[:, :1024].to(dev)
    if ids.shape[1] < 2:
        return 0.0
    with torch.no_grad():
        return float(m["lm"](ids, labels=ids).loss)


def text_feats(r):
    """Features of one transcript record (output of transcribe.transcribe)."""
    t = r["text"]
    words = re.findall(r"[A-Za-z']+", t.lower())
    n = max(len(words), 1)
    sents = [s for s in re.split(r"[.!?]+", t) if s.strip()]
    times = r["word_times"]
    dur = (times[-1][1] - times[0][0]) if times else 1.0
    gaps = [b[0] - a[1] for a, b in zip(times, times[1:])]
    matches = [m for m in _models()["tool"].check(t) if m.rule_id not in IGNORE]
    cats = Counter(m.category for m in matches)
    f = {
        "n_words": len(words), "wps": len(words) / max(dur, 1e-3),
        "n_sents": len(sents), "mean_sent_len": n / max(len(sents), 1),
        "ttr": len(set(words)) / n, "long_word_ratio": np.mean([len(w) > 6 for w in words]) if words else 0,
        "filler_rate": 100 * sum(w in FILLERS for w in words) / n,
        "pause_mean": np.mean(gaps) if gaps else 0, "long_pause_rate": 100 * sum(g > 0.5 for g in gaps) / n,
        "avg_logprob": r["avg_logprob"], "no_speech_prob": r["no_speech_prob"],
        "compression_ratio": r["compression_ratio"],
        "wprob_mean": np.mean(r["word_probs"]) if r["word_probs"] else 0,
        "wprob_low_rate": np.mean(np.array(r["word_probs"]) < 0.5) if r["word_probs"] else 0,
        "gpt2_nll": perplexity(t),
        "err_rate": 100 * len(matches) / n,
    }
    for c in CATS:
        f[f"err_{c.lower()}"] = 100 * cats.get(c, 0) / n
    return f


if __name__ == "__main__":
    rows = {json.loads(l)["key"]: json.loads(l) for p in ["features/transcripts.jsonl"] + glob.glob("features/transcripts.part*.jsonl")
            if os.path.exists(p) for l in open(p)}
    done = set(pd.read_csv(OUT).key) if os.path.exists(OUT) else set()
    rows = [r for k, r in sorted(rows.items()) if k not in done]
    out = []

    def flush():
        global out
        if out:
            pd.DataFrame(out).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False); out = []

    for i, r in enumerate(rows):
        out.append({"key": r["key"], "split": r["split"], "filename": r["filename"], **text_feats(r)})
        if len(out) == 50:
            flush(); print(i + 1, flush=True)
    flush()
    print("saved", len(done) + len(rows))
