"""Mean-pooled self-supervised speech embeddings (WavLM) per layer.

Saves features/wavlm.npz with array [N, L+1, H] (mean over time for every
hidden layer) so we can pick the best layer by cross-validation.
"""
import os, sys
from math import gcd
import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoModel

DATA = "Dataset_Final"
CHUNK = 16000 * 10
dev = "mps" if torch.backends.mps.is_available() else "cpu"
_M = {}


def embed(path, model_name="microsoft/wavlm-base-plus"):
    """[L+1, H] array: every hidden layer mean-pooled over time, for one audio file."""
    if _M.get("name") != model_name:
        _M.update(name=model_name, fe=AutoFeatureExtractor.from_pretrained(model_name),
                  model=AutoModel.from_pretrained(model_name).to(dev).eval())
    fe, model = _M["fe"], _M["model"]
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:  # WavLM expects 16 kHz (every competition clip already is)
        g = gcd(sr, 16000); wav, sr = resample_poly(wav, 16000 // g, sr // g).astype(np.float32), 16000
    # 10 s chunks keep attention memory small; frame-weighted mean over chunks
    x = fe(wav, sampling_rate=sr, return_tensors="pt").input_values[0]
    acc, nfr = 0, 0
    for s in range(0, len(x), CHUNK):
        c = x[s:s + CHUNK]
        if len(c) < sr:  # skip <1 s tail
            continue
        with torch.no_grad():
            hs = model(c[None].to(dev), output_hidden_states=True).hidden_states
        acc = acc + torch.stack([h[0].sum(0) for h in hs]).cpu(); nfr += hs[0].shape[1]
    return (acc / nfr).numpy().astype(np.float32)


if __name__ == "__main__":
    MODEL = sys.argv[1] if len(sys.argv) > 1 else "microsoft/wavlm-base-plus"
    name = MODEL.split("/")[-1]
    CKPT = f"features/{name}.partial.npz"  # resume after interruption
    keys, feats = [], []
    if os.path.exists(CKPT):
        z = np.load(CKPT); keys, feats = list(z["keys"]), list(z["X"])
    for split in ("train", "test"):
        for fn in sorted(os.listdir(f"{DATA}/{split}")):
            if not fn.endswith(".wav") or f"{split}/{fn}" in keys:
                continue
            feats.append(embed(f"{DATA}/{split}/{fn}", MODEL))
            keys.append(f"{split}/{fn}")
            if len(keys) % 100 == 0:
                np.savez(CKPT, keys=np.array(keys), X=np.stack(feats))
                print(len(keys), flush=True)

    np.savez(f"features/{name}.npz", keys=np.array(keys), X=np.stack(feats))
    os.path.exists(CKPT) and os.remove(CKPT)
    print("saved", np.stack(feats).shape)
