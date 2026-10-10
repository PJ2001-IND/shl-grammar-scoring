"""Final model: several base regressors + a non-negative linear stack.

Base models (each evaluated with repeated 5-fold CV, out-of-fold predictions):
  A. Ridge on WavLM mean-pooled embeddings (upper 3/4 of layers) - acoustic/fluency cues
  A3. RBF-kernel SVR on the same WavLM-large embedding - non-linear acoustic patterns
  B. Ridge on interpretable features (grammar-error rates, LLM rubric
     score, fluency, prosody, ASR confidence)
  C. Ridge on sentence-transformer embeddings of the transcript - lexical/syntactic content
  D. Gradient-boosted trees on B features - non-linear interactions
  E. Ridge on rubric-conditioned LLM hidden states (best depth chosen by CV)
  F. Fine-tuned DeBERTa-v3 regressor on the transcript (OOF from src/finetune_text.py)
Stack: non-negative least squares on OOF predictions (+ intercept), so the final
score is a weighted average of base models that is easy to explain.
"""
import os
import numpy as np, pandas as pd
from scipy.optimize import nnls
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR

D = "Dataset_Final"
N_SPLITS, N_REPEATS, SEED = 5, 3, 42


def load_split():
    tr = pd.read_csv(f"{D}/train.csv"); te = pd.read_csv(f"{D}/test.csv")
    return tr, te, ("train/" + tr.filename).tolist(), ("test/" + te.filename).tolist()


HNR_GATE = 0.0  # dB


def quality_ok(keys):
    """Audio-quality gate. Clips with a negative harmonics-to-noise ratio (more noise than
    voice) are corrupted recordings: in train they are exactly the 37 score-0 clips (one
    contiguous batch, audio_5037-5073) and no test clip is one. Supervised models are
    trained on intact clips only; gated clips are predicted 0."""
    hnr = pd.read_csv("features/prosody_feats.csv").set_index("key").loc[keys, "hnr"].values
    return hnr > HNR_GATE


def align(keys_all, X, keys):
    idx = {k: i for i, k in enumerate(keys_all)}
    return X[[idx[k] for k in keys]]


def feature_table(paths=("features/text_feats.csv", "features/prosody_feats.csv",
                         "features/llm_Qwen2.5-7B-Instruct-4bit.csv")):
    # Not used: features/ling_feats.csv (spaCy syntax) did not improve CV (notebook §9); CoEdIT GEC edit
    # rates were dropped because the CoEdIT weights are CC-BY-NC (rule 6c requires commercial-use licences).
    """Join all interpretable per-clip feature CSVs on `key` (skips missing files)."""
    dfs = [pd.read_csv(p).set_index("key") for p in paths if os.path.exists(p)]
    df = pd.concat(dfs, axis=1)
    return df.drop(columns=[c for c in ("split", "filename") if c in df]).fillna(0)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 30)))


def svr():
    """RBF-kernel support-vector regression: a non-linear learner for the speech embeddings."""
    return make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.1, gamma="scale"))


def gbm():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=15,
                                         l2_regularization=1.0, max_features=0.7, random_state=SEED)


def cv_oof(make, X, y, Xte):
    """Repeated K-fold: returns OOF predictions (averaged over repeats) and test
    predictions (averaged over all fold models)."""
    oof = np.zeros((N_REPEATS, len(y))); te = np.zeros(len(Xte)); n = 0
    for i, (a, b) in enumerate(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(X)):
        m = make().fit(X[a], y[a])
        oof[i // N_SPLITS, b] = m.predict(X[b]); te += m.predict(Xte); n += 1
    return oof.mean(0), te / n


def stack(oofs, tes, y):
    """Non-negative weights + intercept fitted on OOF predictions."""
    A = np.column_stack(oofs); mu = y.mean()
    w, _ = nnls(A - A.mean(0), y - mu)
    b = mu - A.mean(0) @ w
    return w, b, np.clip(A @ w + b, 0, 5), np.clip(np.column_stack(tes) @ w + b, 0, 5)


def stack_cv(oofs, y):
    """Honest stack estimate: stack weights refitted inside each fold (nested), so the
    reported stack RMSE does not benefit from fitting weights on the evaluated rows."""
    A = np.column_stack(oofs); out = np.zeros((N_REPEATS, len(y)))
    for i, (a, b) in enumerate(RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED + 1).split(A)):
        w, b0, _, p = stack([A[a][:, j] for j in range(A.shape[1])], [A[b][:, j] for j in range(A.shape[1])], y[a])
        out[i // N_SPLITS, b] = p
    return out.mean(0)


def report(name, p, y):
    p = np.clip(p, 0, 5)
    print(f"{name:28s} RMSE {rmse(p, y):.4f}   Pearson {pearsonr(p, y)[0]:.4f}")
