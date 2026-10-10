"""Reviewer-style checks for the whole project:  `pytest -q tests/`

1. the data matches the competition spec (counts, columns, audio format, label scale)
2. submission.csv is a valid Kaggle submission
3. train/ and test/ clips with the same file name are never confused (212 names are shared)
4. the audio-quality gate does what the report claims
5. no label leakage: with shuffled labels the model is no better than predicting the mean
6. feature extraction survives bad inputs (silence, noise, very short, 44.1 kHz stereo)
7. CV is deterministic; the notebook meets the submission requirements
8. the single-file scorer reproduces the notebook (SLOW=1 also runs raw audio -> score end to end)
Tests needing the (non-redistributable) competition data or cached features are skipped without them.
"""
import json, os, sys
import numpy as np, pandas as pd, pytest, soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
from model import load_split, quality_ok, feature_table, cv_oof, gbm, rmse  # noqa: E402

D = "Dataset_Final"
need_data = pytest.mark.skipif(not os.path.exists(f"{D}/train.csv"), reason="competition data not present")
need_feats = pytest.mark.skipif(not os.path.exists("features/prosody_feats.csv"), reason="features not extracted")


# 1. data spec ------------------------------------------------------------------------------------
@need_data
def test_data_matches_competition_spec():
    tr, te, _, _ = load_split()
    assert list(tr.columns) == ["filename", "label"] and list(te.columns) == ["filename", "label"]
    assert len(tr) == 769 and len(te) == 216                         # "769 samples" / "216 samples"
    assert tr.filename.is_unique and te.filename.is_unique
    assert tr.label.between(0, 5).all() and ((tr.label * 2) % 1 == 0).all()   # 0-5 in 0.5 steps
    for split, df in (("train", tr), ("test", te)):
        for f in df.filename:
            i = sf.info(f"{D}/{split}/{f}")
            assert (i.samplerate, i.channels) == (16000, 1), f


# 2. submission -----------------------------------------------------------------------------------
@need_data
def test_submission_is_valid():
    s, te = pd.read_csv("submission.csv"), pd.read_csv(f"{D}/test.csv")
    assert list(s.columns) == list(pd.read_csv(f"{D}/sample_submission.csv").columns) == ["filename", "label"]
    assert len(s) == 216 and s.filename.is_unique and set(s.filename) == set(te.filename)
    assert pd.api.types.is_float_dtype(s.label) and np.isfinite(s.label).all() and s.label.between(0, 5).all()
    assert s.label.std() > 0.3                                         # not a constant / degenerate prediction


# 3. train/test key separation --------------------------------------------------------------------
@need_data
@need_feats
def test_shared_filenames_are_kept_apart():
    tr, te, tr_keys, te_keys = load_split()
    shared = set(tr.filename) & set(te.filename)
    assert len(shared) > 200                     # the trap: most test names also exist in train/
    T = {json.loads(l)["key"]: json.loads(l)["text"] for l in open("features/transcripts.jsonl")}
    assert set(tr_keys + te_keys) <= set(T)      # every clip transcribed under its split-prefixed key
    same = sum(T[f"train/{f}"] == T[f"test/{f}"] for f in shared)
    assert same == 0                             # different recordings -> different transcripts
    ft = feature_table()
    assert set(tr_keys + te_keys) <= set(ft.index)


# 4. quality gate ---------------------------------------------------------------------------------
@need_data
@need_feats
def test_quality_gate():
    tr, te, tr_keys, te_keys = load_split()
    bad = ~quality_ok(tr_keys)
    assert bad.sum() == 37 and (tr.label[bad] == 0).all() and (tr.label[~bad] > 0).all()
    assert quality_ok(te_keys).all()
    s = pd.read_csv("submission.csv")
    assert (s.label > 0).all()                    # no intact test clip predicted as "corrupted"


# 5. leakage --------------------------------------------------------------------------------------
@need_data
@need_feats
def test_no_label_leakage_permutation():
    tr, _, tr_keys, _ = load_split()
    ok = quality_ok(tr_keys); y = tr.label.values[ok]
    X = feature_table().loc[tr_keys].values[ok]
    base = rmse(np.full_like(y, y.mean()), y)
    real = rmse(np.clip(cv_oof(gbm, X, y, X[:2])[0], 0, 5), y)
    perm = rmse(np.clip(cv_oof(gbm, X, np.random.default_rng(0).permutation(y), X[:2])[0], 0, 5), y)
    assert real < 0.7 * base                      # features carry real signal ...
    assert perm > 0.97 * base                     # ... and none of it survives shuffling the labels


# 6. robustness of feature extraction to bad audio -------------------------------------------------
@pytest.fixture
def bad_audio(tmp_path):
    rng, sr = np.random.default_rng(0), 16000
    t = np.arange(int(3 * sr)) / sr
    voiced = 0.3 * np.sin(2 * np.pi * 150 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))  # syllable-like AM tone
    clips = {"silence.wav": (np.zeros(sr * 50), sr),
             "noise.wav": (0.3 * rng.standard_normal(sr * 50), sr),
             "short.wav": (voiced, sr),
             "stereo44k.wav": (np.stack([np.resize(voiced, 44100 * 10)] * 2, 1), 44100)}
    for n, (x, r) in clips.items():
        sf.write(tmp_path / n, x, r)
    return tmp_path


def test_prosody_survives_bad_audio(bad_audio, monkeypatch):
    import prosody_features as P
    monkeypatch.setattr(P, "DATA", str(bad_audio))
    out = {n: P.feats(n) for n in os.listdir(bad_audio)}
    for n, f in out.items():
        vals = np.array([v for k, v in f.items() if k != "key"], float)
        assert np.isfinite(np.nan_to_num(vals, nan=0.0)).all(), n
    assert not out["noise.wav"]["hnr"] > 0        # pure noise must fall below the gate (HNR <= 0 or undefined)
    assert out["short.wav"]["duration"] == pytest.approx(3, abs=0.05)


# 7. determinism + notebook requirements -----------------------------------------------------------
def test_cv_is_deterministic():
    rng = np.random.default_rng(1); X = rng.normal(size=(120, 5)); y = X[:, 0] + rng.normal(size=120)
    a, b = cv_oof(gbm, X, y, X[:3]), cv_oof(gbm, X, y, X[:3])
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_notebook_meets_requirements():
    nb = json.load(open("grammar_scoring.ipynb"))
    md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown").lower()
    for topic in ("approach", "preprocessing", "pipeline architecture", "evaluation results", "rubric", "single-file"):
        assert topic in md, topic                 # report sections the task asks for
    outs = [o for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", [])]
    assert not [o for o in outs if o.get("output_type") == "error"]
    assert sum("image/png" in o.get("data", {}) for o in outs) >= 5          # visualisations
    text = "\n".join("".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", "")) for o in outs)
    assert "Training data" in text and "RMSE" in text and "Pearson" in text   # compulsory training RMSE


# 8. single-file scorer ---------------------------------------------------------------------------
need_scorer = pytest.mark.skipif(not os.path.exists("models/scorer.joblib"), reason="run the notebook first")


@need_feats
@need_scorer
def test_scorer_bundle_matches_training_features():
    """The saved scorer, fed the cached training-pipeline features, must agree with the notebook's
    (fold-averaged) test predictions; a mismatch would mean features/columns are misaligned."""
    import joblib
    from model import align
    S = joblib.load("models/scorer.joblib"); ft = feature_table()
    _, te, _, te_keys = load_split()
    w = np.load("features/wavlm-large.npz"); z = np.load("features/llm_Qwen2.5-7B-Instruct-4bit_emb.npz")
    xb = ft.loc[te_keys, S["features"]].values
    xw = align(w["keys"], w["X"], te_keys)[:, S["wavlm_lo"]:].reshape(len(te_keys), -1)
    X = {"A2_wavlm_large_ridge": xw, "A3_wavlm_large_svr": xw,
         "B_features_ridge": xb, "D_features_gbm": xb,
         "E_llm_states_ridge": align(z["keys"], z["X"], te_keys)[:, S["llm_layer"]].astype(np.float32)}
    g = np.load("features/whisper_enc.npz")
    X["G_whisper_enc_svr"] = align(g["keys"], g["X"], te_keys)[:, S["whisper_lo"]:].reshape(len(te_keys), -1).astype(np.float32)
    p = np.clip(np.column_stack([S["models"][n].predict(X[n]) for n in S["names"]]) @ S["weights"] + S["intercept"], 0, 5)
    ref = np.array([S["test_pred"][f] for f in te.filename])
    assert np.corrcoef(p, ref)[0, 1] > 0.98 and np.abs(p - ref).mean() < 0.06


@pytest.mark.skipif(not os.environ.get("SLOW"), reason="end-to-end model run (~2 min); set SLOW=1")
@need_data
@need_scorer
def test_predict_end_to_end(bad_audio):
    """Raw audio -> score through src/predict.py, including a very short clip and pure noise."""
    import joblib, predict
    S = joblib.load("models/scorer.joblib")
    for f in ("audio_107.wav", "audio_48.wav"):          # 6.5 s clip and a normal clip
        s, d = predict.score(f"{D}/test/{f}", S)
        assert not d["gated"] and abs(s - S["test_pred"][f]) < 0.15, (f, s)
    assert predict.score(str(bad_audio / "noise.wav"), S)[0] == 0.0
    s, _ = predict.score(str(bad_audio / "stereo44k.wav"), S)   # non-16 kHz input is resampled, not a crash
    assert 0 <= s <= 5
