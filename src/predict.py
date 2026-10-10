"""Score audio files with the trained grammar scorer:  audio file in -> score in [0, 5] out.

Usage: python src/predict.py clip.wav [more.wav ...] [--csv out.csv]

Needs models/scorer.joblib (written by the notebook, section 8) and the public pretrained models;
not the competition data. Per clip: Praat prosody (+ quality gate) -> Whisper transcript ->
LanguageTool/GPT-2 text features + rubric-prompted Qwen judge/hidden states -> WavLM-large
and Whisper-encoder embeddings -> the refitted base models -> non-negative stack -> clip to [0, 5].
"""
import argparse, os, sys
import numpy as np, pandas as pd, joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prosody_features, transcribe, text_features, llm_judge, audio_embed, whisper_embed  # noqa: E402

BUNDLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "scorer.joblib")


def score(path, S):
    """Returns (score, details) for one audio file using the loaded bundle S."""
    path = os.path.abspath(path)
    prosody_features.DATA = os.path.dirname(path)
    pros = prosody_features.feats(os.path.basename(path))
    if not pros["hnr"] > S["hnr_gate"]:   # corrupted / non-speech recording
        return 0.0, {"gated": True, "hnr": pros["hnr"]}
    tr = transcribe.transcribe(path)
    lp, emb = llm_judge.forward(tr["text"] or "(no speech)")
    feats = {**text_features.text_feats(tr), **pros, **llm_judge.judge_table(lp).iloc[0].to_dict()}
    xb = np.nan_to_num(np.array([[feats.get(c, 0.0) for c in S["features"]]], float))  # training used fillna(0)
    w = audio_embed.embed(path, S["wavlm"])[S["wavlm_lo"]:].reshape(1, -1)
    X = {"A2_wavlm_large_ridge": w, "A3_wavlm_large_svr": w,
         "B_features_ridge": xb, "D_features_gbm": xb,
         "E_llm_states_ridge": emb[S["llm_layer"]][None].astype(np.float16).astype(np.float32),  # fp16 as stored
         "G_whisper_enc_svr": whisper_embed.embed(path)[S["whisper_lo"]:].astype(np.float16).astype(np.float32).reshape(1, -1)}
    base = np.array([S["models"][n].predict(X[n])[0] for n in S["names"]])
    g = int(44 < pros["duration"] < 46)                         # recording batch, as model.batch
    s = float(np.clip(base @ S["weights"][g] + S["intercept"][g], 0, 5))
    return s, {"gated": False, "hnr": pros["hnr"], "transcript": tr["text"], "llm_expected": feats["llm_expected"],
               **dict(zip(S["names"], base.round(3)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("audio", nargs="+"); ap.add_argument("--csv")
    a = ap.parse_args()
    if not os.path.exists(BUNDLE):
        sys.exit(f"{BUNDLE} not found: run the notebook (or ./run_all.sh) first")
    S = joblib.load(BUNDLE)
    rows = []
    for p in a.audio:
        s, d = score(p, S)
        rows.append({"filename": os.path.basename(p), "label": s})
        print(f"{p}: {s:.2f}" + ("   [quality gate: corrupted recording]" if d["gated"] else
                                 f"   (LLM rubric judge {d['llm_expected']:.2f}) \"{d['transcript'][:80]}...\""), flush=True)
    if a.csv:
        pd.DataFrame(rows).to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
