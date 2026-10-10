"""Acoustic fluency / prosody features with Praat (parselmouth), CPU only.

Independent of the ASR, so it still works when Whisper "cleans up" disfluent speech.
1. Speech rate via syllable nuclei (simplified de Jong & Wempe 2009): intensity peaks
   that are voiced, above a silence threshold and separated by a >=2 dB dip.
2. Pauses: silent intervals >= 0.25 s (count, mean length, share of total time).
3. Pitch (semitones re 100 Hz): spread and range - monotone vs. expressive speech.
4. Voice quality: harmonics-to-noise ratio, jitter, shimmer.
Writes features/prosody_feats.csv
"""
import os
import numpy as np, pandas as pd
import parselmouth
from parselmouth.praat import call
from multiprocessing import Pool

DATA = "Dataset_Final"
MIN_PAUSE, DIP_DB = 0.25, 2.0


def feats(key):
    snd = parselmouth.Sound(f"{DATA}/{key}")
    dur = snd.duration
    inten = snd.to_intensity(minimum_pitch=50, time_step=0.01)
    iv, it = inten.values[0], inten.xs()
    thr = max(np.percentile(iv, 99) - 25, np.median(iv) - 2)  # silence threshold (dB)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    voiced_at = lambda t: pitch.get_value_at_time(t) == pitch.get_value_at_time(t)  # not NaN

    # syllable nuclei: local maxima above threshold, separated from previous peak by a dip
    peaks = [i for i in range(1, len(iv) - 1) if iv[i] > thr and iv[i] >= iv[i - 1] and iv[i] > iv[i + 1]]
    nuclei, last = [], None
    for i in peaks:
        if last is not None and iv[last:i + 1].min() > min(iv[last], iv[i]) - DIP_DB:
            if iv[i] > iv[last]:
                nuclei[-1] = i; last = i
            continue
        if voiced_at(it[i]):
            nuclei.append(i); last = i

    # pauses: runs of below-threshold frames between first and last speech frame
    sil = iv < thr
    sp = np.where(~sil)[0]
    a, b = (sp[0], sp[-1]) if len(sp) else (0, len(iv) - 1)
    lo, hi = it[a], it[b]
    runs, start = [], None
    for i in range(a, b + 2):
        s = i <= b and sil[i]
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append((i - start) * 0.01); start = None
    pauses = [r for r in runs if r >= MIN_PAUSE]
    phon = max(hi - lo - sum(pauses), 1e-3)

    st = 12 * np.log2(f0[f0 > 0] / 100) if (f0 > 0).sum() > 10 else np.zeros(1)
    pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    hnr = call(snd.to_harmonicity_cc(), "Get mean", 0, 0)
    jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shim = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    n = len(nuclei)
    return {
        "key": key, "duration": dur, "speech_time": hi - lo,
        "speech_rate": n / max(hi - lo, 1e-3), "articulation_rate": n / phon,
        "phonation_ratio": phon / max(hi - lo, 1e-3), "n_syllables": n,
        "n_pauses_per_min": 60 * len(pauses) / max(hi - lo, 1e-3),
        "pause_mean_len": np.mean(pauses) if pauses else 0.0,
        "pause_long_rate": 60 * sum(p > 1.0 for p in pauses) / max(hi - lo, 1e-3),
        "syll_per_run": n / (len(pauses) + 1),  # mean length of run between pauses
        "pitch_sd_st": float(np.std(st)), "pitch_range_st": float(np.percentile(st, 95) - np.percentile(st, 5)),
        "intensity_sd": float(np.std(iv[~sil])) if (~sil).any() else 0.0,
        "voiced_ratio": float((f0 > 0).mean()),
        "hnr": hnr if np.isfinite(hnr) else 0.0,
        "jitter": jit if np.isfinite(jit) else 0.0, "shimmer": shim if np.isfinite(shim) else 0.0,
    }


if __name__ == "__main__":
    keys = [f"{s}/{f}" for s in ("train", "test") for f in sorted(os.listdir(f"{DATA}/{s}")) if f.endswith(".wav")]
    with Pool(4) as p:
        out = p.map(feats, keys, chunksize=8)
    pd.DataFrame(out).to_csv("features/prosody_feats.csv", index=False)
    print("saved", len(out))
