"""Whisper encoder embeddings: the ASR model's own acoustic representation of the speech.

Whisper's encoder was trained to map speech to text across many accents and proficiency levels, so its
states encode pronunciation, fluency and how "transcribable" the speech is. Each clip is cut into 30 s
windows (the encoder's input size); hidden states are mean-pooled over the frames that contain audio
(not the zero padding) at every 4th block, frame-weighted across windows.
Usage: python src/whisper_embed.py      Writes features/whisper_enc.npz  ([N, 8, 1280], fp16)
"""
import os
import numpy as np, soundfile as sf
import mlx.core as mx, mlx.nn as nn
from mlx_whisper.load_models import load_model
from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_SAMPLES, N_SAMPLES_PER_TOKEN

MODEL = "mlx-community/whisper-large-v3-turbo"
DATA = "Dataset_Final"
OUT = "features/whisper_enc.npz"
mx.set_cache_limit(1 << 30)
_M = {}


def _model():
    if not _M:
        m = load_model(MODEL, dtype=mx.float16)
        _M.update(enc=m.encoder, n_mels=m.dims.n_mels,
                  keep=list(range(4, len(m.encoder.blocks) + 1, 4)))  # blocks 4, 8, ..., 32
    return _M


def embed(path):
    """[8, 1280] array: encoder states at every 4th block, mean-pooled over the audio frames."""
    M = _model(); enc = M["enc"]
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if sr != 16000:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr, 16000); wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
    acc, nfr = 0, 0
    for s in range(0, len(wav), N_SAMPLES):
        seg = wav[s:s + N_SAMPLES]
        if len(seg) < 16000 and s > 0:  # skip <1 s tail
            continue
        n = max(1, int(np.ceil(len(seg) / N_SAMPLES_PER_TOKEN)))  # encoder frames holding real audio
        mel = log_mel_spectrogram(pad_or_trim(mx.array(seg)), n_mels=M["n_mels"])[None].astype(mx.float16)
        x = nn.gelu(enc.conv1(mel)); x = nn.gelu(enc.conv2(x)); x = x + enc._positional_embedding
        pooled = []
        for i, block in enumerate(enc.blocks, 1):
            x, _, _ = block(x)
            if i in M["keep"]:
                pooled.append((enc.ln_post(x) if i == len(enc.blocks) else x)[0, :n].astype(mx.float32).sum(0))
        p = mx.stack(pooled); mx.eval(p)
        acc = acc + np.array(p); nfr += n
    return (acc / nfr).astype(np.float32)


if __name__ == "__main__":
    keys, X = [], []
    for split in ("train", "test"):
        for fn in sorted(os.listdir(f"{DATA}/{split}")):
            if fn.endswith(".wav"):
                X.append(embed(f"{DATA}/{split}/{fn}").astype(np.float16)); keys.append(f"{split}/{fn}")
                if len(keys) % 100 == 0:
                    print(len(keys), flush=True)
    np.savez(OUT, keys=np.array(keys), X=np.stack(X), layers=np.array(_model()["keep"]))
    print("saved", np.stack(X).shape)
