"""Fine-tune DeBERTa-v3 to regress the grammar score from the transcript.

Trained on intact recordings only (see model.quality_ok); gated rows of the saved
arrays are unused. 5-fold CV: each fold model predicts its held-out fold (out-of-fold predictions for
stacking / honest evaluation) and the test set (averaged over folds).
Loss: MSE on the label scaled to [0, 1]; mean pooling over tokens; layer-wise LR.
Writes features/deberta_oof.npy, deberta_test.npy and deberta_train.npy (fold-averaged
predictions on the training set, only used to report the compulsory training RMSE)
"""
import json, os, sys
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")  # cap the MPS memory pool on a 16 GB Mac
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")   # (must not exceed the high ratio)
import numpy as np, pandas as pd, torch
from torch import nn
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup

MODEL = sys.argv[1] if len(sys.argv) > 1 else "microsoft/deberta-v3-base"
EPOCHS, BS, LR, MAXLEN, SEED = 6, 8, 3e-5, 320, 42
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

D = "Dataset_Final"
tr = pd.read_csv(f"{D}/train.csv"); te = pd.read_csv(f"{D}/test.csv")
T = {json.loads(l)["key"]: json.loads(l)["text"] for l in open("features/transcripts.jsonl")}
Xtr = [T["train/" + f] for f in tr.filename]; Xte = [T["test/" + f] for f in te.filename]
y = tr.label.values.astype(np.float32) / 5.0
sys.path.insert(0, "src")
from model import quality_ok
ok = np.where(quality_ok(("train/" + tr.filename).tolist()))[0]  # train on intact recordings only
tok = AutoTokenizer.from_pretrained(MODEL)


class Reg(nn.Module):
    def __init__(self):
        super().__init__()
        # float32 weights: transformers loads the checkpoint dtype (fp16) by default, and
        # fp16 training without loss scaling diverges to NaN
        self.enc = AutoModel.from_pretrained(MODEL, dtype=torch.float32)
        self.enc.gradient_checkpointing_enable()  # recompute activations: ~half the memory, same result
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(self.enc.config.hidden_size, 1))

    def forward(self, ids, mask):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        return self.head((h * m).sum(1) / m.sum(1)).squeeze(-1)


def batches(texts, labels=None, shuffle=False):
    idx = np.random.permutation(len(texts)) if shuffle else np.arange(len(texts))
    for i in range(0, len(idx), BS):
        b = idx[i:i + BS]
        x = tok([texts[j] for j in b], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        yield x.input_ids.to(dev), x.attention_mask.to(dev), (torch.tensor(labels[b]).to(dev) if labels is not None else None)


def predict(m, texts):
    m.eval(); out = []
    with torch.no_grad():
        for ids, mask, _ in batches(texts):
            out.append(m(ids, mask).float().cpu().numpy())
    return np.concatenate(out)


oof = np.zeros(len(Xtr)); test = np.zeros(len(Xte)); train = np.zeros(len(Xtr))
for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(ok)):
    a, b = ok[a], ok[b]  # fold positions -> training-set rows
    m = Reg().to(dev)
    params = [{"params": m.enc.parameters(), "lr": LR}, {"params": m.head.parameters(), "lr": 1e-3}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    steps = EPOCHS * int(np.ceil(len(a) / BS))
    sch = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
    ta = [Xtr[i] for i in a]; tb = [Xtr[i] for i in b]
    for ep in range(EPOCHS):
        m.train()
        for ids, mask, yy in batches(ta, y[a], shuffle=True):
            loss = nn.functional.mse_loss(m(ids, mask), yy)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step()
        p = predict(m, tb)
        r = float(np.sqrt(np.mean((np.clip(p, 0, 1) * 5 - y[b] * 5) ** 2)))
        print(f"fold {k} epoch {ep} val RMSE {r:.4f}", flush=True)
    # use the final epoch (fixed schedule; no peeking at the validation fold for model selection)
    oof[b] = p * 5; test += predict(m, Xte) * 5 / 5; train += predict(m, Xtr) * 5 / 5
    del m, opt; torch.mps.empty_cache() if dev == "mps" else None

np.save("features/deberta_oof.npy", oof); np.save("features/deberta_test.npy", test); np.save("features/deberta_train.npy", train)
yy = tr.label.values[ok]; o = oof[ok]
print("DeBERTa OOF RMSE (intact clips)", np.sqrt(np.mean((np.clip(o, 0, 5) - yy) ** 2)), "r", np.corrcoef(o, yy)[0, 1])
