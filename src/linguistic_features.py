"""Rubric-aligned linguistic features from the transcripts (CPU only, fast).

The SHL rubric rewards "complex grammar structures", penalises "incomplete sentences"
and credits speakers who "correct themselves". These features measure exactly that:
1. Syntactic complexity (spaCy dependency parse): parse-tree depth, clauses and
   subordinate/relative clauses per sentence, passive voice, coordination, verb forms.
2. Disfluency / self-repair: immediate word and phrase repetitions ("I I", "in the in the"),
   fillers, and sentences without a main verb (fragments).
3. Lexical sophistication (wordfreq): mean Zipf frequency and share of rare words.
4. ASR coverage: words recognised by Whisper per syllable detected by Praat - unclear
   speech makes Whisper drop words, so a low ratio flags hard-to-understand speech.
Writes features/ling_feats.csv
"""
import json
import numpy as np, pandas as pd
import spacy
from wordfreq import zipf_frequency

nlp = spacy.load("en_core_web_sm")
P = pd.read_csv("features/prosody_feats.csv").set_index("key")
FILLERS = {"um", "uh", "erm", "er", "ah", "hmm", "mm", "uhm", "like"}
CLAUSE = {"ccomp", "xcomp", "advcl", "acl", "relcl", "csubj", "csubjpass"}


def depth(tok):
    d = 0
    while tok.head.i != tok.i:  # compare indices: spaCy returns a new Token object on each .head
        tok, d = tok.head, d + 1
    return d


def feats(r):
    text = r["text"] or ""
    doc = nlp(text)
    toks = [t for t in doc if t.is_alpha]
    w = [t.lower_ for t in toks]
    n = max(len(w), 1)
    sents = [s for s in doc.sents if any(t.is_alpha for t in s)]
    ns = max(len(sents), 1)
    # immediate repetitions of 1-3 word sequences ("I I", "in the in the")
    reps = sum(1 for k in (1, 2, 3) for i in range(len(w) - 2 * k + 1) if w[i:i + k] == w[i + k:i + 2 * k])
    zipf = np.array([zipf_frequency(x, "en") for x in w]) if w else np.zeros(1)
    verbs = [t for t in doc if t.pos_ in ("VERB", "AUX")]
    return {
        "key": r["key"],
        # syntactic complexity
        "parse_depth_mean": float(np.mean([max(depth(t) for t in s) for s in sents])) if sents else 0.0,
        "clauses_per_sent": sum(t.dep_ in CLAUSE for t in doc) / ns,
        "subord_per_sent": sum(t.dep_ in ("advcl", "mark") for t in doc) / ns,
        "relcl_per_sent": sum(t.dep_ == "relcl" for t in doc) / ns,
        "coord_per_sent": sum(t.dep_ == "conj" for t in doc) / ns,
        "passive_rate": 100 * sum(t.dep_ in ("nsubjpass", "auxpass") for t in doc) / n,
        "verb_form_variety": len({t.tag_ for t in verbs}) / max(len(verbs), 1) if verbs else 0.0,
        "verbs_per_100w": 100 * len(verbs) / n,
        "fragment_ratio": sum(not any(t.pos_ in ("VERB", "AUX") for t in s) for s in sents) / ns,
        # disfluency / self-repair
        "repetition_rate": 100 * reps / n,
        "filler_rate2": 100 * sum(x in FILLERS for x in w) / n,
        # lexical sophistication
        "zipf_mean": float(zipf.mean()), "rare_word_ratio": float((zipf < 4.0).mean()),
        "lexical_density": sum(t.pos_ in ("NOUN", "VERB", "ADJ", "ADV") for t in toks) / n,
        # ASR coverage of the audio
        "words_per_syllable": len(w) / max(P.loc[r["key"], "n_syllables"], 1),
    }


rows = [json.loads(l) for l in open("features/transcripts.jsonl")]
pd.DataFrame([feats(r) for r in rows]).to_csv("features/ling_feats.csv", index=False)
print("saved", len(rows))
