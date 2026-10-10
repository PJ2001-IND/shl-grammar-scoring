"""Intermediate submission from the cheap features only (text + prosody), produced
as soon as transcription is done, while the heavier models are still running.

Ridge + gradient-boosted trees on the interpretable features, NNLS stack, with the
audio-quality gate (model.quality_ok).
Writes submission_v2_text_prosody.csv
"""
import sys, warnings
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import pandas as pd
from model import load_split, quality_ok, feature_table, ridge, gbm, cv_oof, stack, stack_cv, report

tr, te, tr_keys, te_keys = load_split()
ok, ok_te = quality_ok(tr_keys), quality_ok(te_keys)  # audio-quality gate
y = tr.label.values[ok]
ft = feature_table(("features/text_feats.csv", "features/prosody_feats.csv"))
Xtr, Xte = ft.loc[tr_keys].values[ok], ft.loc[te_keys].values
r, g = cv_oof(ridge, Xtr, y, Xte), cv_oof(gbm, Xtr, y, Xte)
w, b, _, pred = stack([r[0], g[0]], [r[1], g[1]], y)
report("text+prosody stack (nested CV, intact clips)", stack_cv([r[0], g[0]], y), y)
pred[~ok_te] = 0.0
pd.DataFrame({"filename": te.filename, "label": pred}).to_csv("submission_v2_text_prosody.csv", index=False)
print("wrote submission_v2_text_prosody.csv", pred.shape)
