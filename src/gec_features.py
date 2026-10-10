"""Grammatical-error-correction (GEC) edit features.

CoEdIT-large (a Flan-T5 model fine-tuned for text editing) rewrites each sentence
with the instruction "Fix grammatical errors in this sentence:". The more words it
has to change, the less grammatical the speaker. Edits are counted on lower-cased,
punctuation-free word sequences so ASR punctuation does not count as an error.
Writes features/gec_feats.csv. Incremental: clips already in the CSV are skipped and
progress is appended every 50 clips, so an interrupted run resumes where it stopped.
"""
import json, os, re, difflib
import pandas as pd, torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

MODEL = "grammarly/coedit-large"
dev = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = T5ForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float16 if dev == "mps" else torch.float32).to(dev).eval()
words = lambda s: re.findall(r"[a-z0-9']+", s.lower())


def correct(sents, bs=16):
    out = []
    for i in range(0, len(sents), bs):
        b = ["Fix grammatical errors in this sentence: " + s for s in sents[i:i + bs]]
        x = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
        with torch.no_grad():
            g = model.generate(**x, max_new_tokens=128, num_beams=1)
        out += tok.batch_decode(g, skip_special_tokens=True)
    return out


OUT = "features/gec_feats.csv"
rows = [json.loads(l) for l in open("features/transcripts.jsonl")]
done = set(pd.read_csv(OUT).key) if os.path.exists(OUT) else set()
rows = [r for r in rows if r["key"] not in done]
CHUNK = 50  # transcripts per GPU round / checkpoint
for c in range(0, len(rows), CHUNK):
    chunk = rows[c:c + CHUNK]
    sents = [[s.strip() for s in re.split(r"(?<=[.!?])\s+", r["text"]) if len(words(s)) > 0] for r in chunk]
    # pool sentences across transcripts and sort by length: full, evenly padded GPU batches
    flat = [(i, s) for i, ss in enumerate(sents) for s in ss]
    order = sorted(range(len(flat)), key=lambda j: len(flat[j][1]))
    fixed_sorted = correct([flat[j][1] for j in order], bs=48)
    fixed = [None] * len(flat)
    for j, f in zip(order, fixed_sorted):
        fixed[j] = f
    per_row = [[] for _ in chunk]
    for (i, _), f in zip(flat, fixed):
        per_row[i].append(f)
    feats = []
    for r, ss, ff in zip(chunk, sents, per_row):
        ins = dele = sub = tot = changed = 0
        for a, b in zip(ss, ff):
            wa, wb = words(a), words(b); tot += len(wa)
            ch = False
            for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=wa, b=wb, autojunk=False).get_opcodes():
                if op == "replace": sub += max(i2 - i1, j2 - j1); ch = True
                elif op == "delete": dele += i2 - i1; ch = True
                elif op == "insert": ins += j2 - j1; ch = True
            changed += ch
        tot = max(tot, 1)
        feats.append({"key": r["key"], "gec_edit_rate": 100 * (ins + dele + sub) / tot,
                      "gec_sub_rate": 100 * sub / tot, "gec_ins_rate": 100 * ins / tot, "gec_del_rate": 100 * dele / tot,
                      "gec_changed_sent_ratio": changed / max(len(ss), 1)})
    pd.DataFrame(feats).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    print("gec", len(done) + c + len(chunk), flush=True)
print("saved", len(done) + len(rows))
