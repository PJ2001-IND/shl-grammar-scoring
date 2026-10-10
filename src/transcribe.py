"""Transcribe all train/test audio with Whisper (MLX, Apple Silicon).

Keeps verbatim-ish output (no prompt cleanup) plus per-segment confidence stats,
since disfluencies and low-confidence speech are themselves signals of proficiency.

Usage: python src/transcribe.py [shard n_shards]   (run several shards in parallel:
decoding is sequential, so one process leaves the GPU mostly idle), then
python src/transcribe.py merge
"""
import glob, json, os, sys
import numpy as np

DATA = "Dataset_Final"
OUT = "features/transcripts.jsonl"
MODEL = "mlx-community/whisper-large-v3-turbo"



def transcribe(path):
    """Whisper transcript + confidence/timing stats for one audio file."""
    import mlx_whisper
    r = mlx_whisper.transcribe(path, path_or_hf_repo=MODEL, language="en",
                               condition_on_previous_text=False, word_timestamps=True)
    segs = r["segments"]
    words = [w for s in segs for w in s.get("words", [])]
    return {
        "text": r["text"].strip(),
        "avg_logprob": float(np.mean([s["avg_logprob"] for s in segs])) if segs else 0.0,
        "no_speech_prob": float(np.mean([s["no_speech_prob"] for s in segs])) if segs else 1.0,
        "compression_ratio": float(np.mean([s["compression_ratio"] for s in segs])) if segs else 0.0,
        "word_probs": [w["probability"] for w in words],
        "word_times": [[w["start"], w["end"]] for w in words],
    }


if __name__ == "__main__":
    PARTS = sorted(glob.glob("features/transcripts.part*.jsonl"))
    if sys.argv[1:] == ["merge"]:
        rows = {}
        for p in [OUT] + PARTS:
            if os.path.exists(p):
                for l in open(p):
                    r = json.loads(l); rows[r["key"]] = r
        with open(OUT, "w") as fo:
            for k in sorted(rows):
                fo.write(json.dumps(rows[k]) + "\n")
        for p in PARTS:
            os.remove(p)
        sys.exit(print("merged", len(rows)))

    shard, n_shards = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 1)
    done = {json.loads(l)["key"] for p in [OUT] + PARTS if os.path.exists(p) for l in open(p)}
    files = [(s, f) for s in ("train", "test") for f in sorted(os.listdir(f"{DATA}/{s}")) if f.endswith(".wav")]
    files = [x for x in files if f"{x[0]}/{x[1]}" not in done][shard::n_shards]
    with open(f"features/transcripts.part{shard}.jsonl" if n_shards > 1 else OUT, "a") as fo:
        for i, (split, fn) in enumerate(files):
            key = f"{split}/{fn}"
            fo.write(json.dumps({"key": key, "split": split, "filename": fn, **transcribe(f"{DATA}/{key}")}) + "\n")
            fo.flush()
            print(i, key, flush=True)
