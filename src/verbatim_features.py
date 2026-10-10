"""Disfluency features from the verbatim transcripts (VERBATIM=1 python src/transcribe.py).

The rubric credits speakers who "correct themselves" (score 5) and penalises "incomplete sentences" (score 2).
Prompted to transcribe verbatim, Whisper keeps the repetitions, false starts and self-corrections that its
default output tidies away, so comparing the two transcripts measures them directly:
- repetition rate: immediate repeats of 1-3 word sequences ("there are, there are")
- false starts: words cut off with a hyphen ("play- the playground")
- fillers ("um", "uh", ...) and the share of words the default transcript dropped (cleanup rate)
Writes features/verbatim_feats.csv
"""
import json, re
import pandas as pd

FILLERS = {"um", "umm", "uh", "uhh", "erm", "er", "ah", "hmm", "mm", "uhm"}
words = lambda t: re.findall(r"[a-z']+-?", t.lower())


def feats(v, o):
    """v: verbatim transcript record, o: default transcript record."""
    wv, wo = words(v["text"]), words(o["text"])
    w = [x.rstrip("-") for x in wv]; n = max(len(w), 1)
    reps = sum(1 for k in (1, 2, 3) for i in range(len(w) - 2 * k + 1) if w[i:i + k] == w[i + k:i + 2 * k])
    dur = (v["word_times"][-1][1] - v["word_times"][0][0]) / 60 if v["word_times"] else 1.0
    return {
        "key": v["key"],
        "vb_repetition_rate": 100 * reps / n,
        "vb_false_start_rate": 100 * sum(x.endswith("-") for x in wv) / n,
        "vb_filler_rate": 100 * sum(x in FILLERS for x in w) / n,
        "vb_cleanup_rate": 100 * max(len(wv) - len(wo), 0) / n,        # words the default transcript dropped
        "vb_words_per_min": len(w) / max(dur, 1e-3),
        "vb_mean_sent_len": n / max(len([s for s in re.split(r"[.!?]+", v["text"]) if s.strip()]), 1),
    }


if __name__ == "__main__":
    O = {json.loads(l)["key"]: json.loads(l) for l in open("features/transcripts.jsonl")}
    V = [json.loads(l) for l in open("features/transcripts_verbatim.jsonl")]
    pd.DataFrame([feats(v, O[v["key"]]) for v in V]).to_csv("features/verbatim_feats.csv", index=False)
    print("saved", len(V))
