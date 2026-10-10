#!/usr/bin/env bash
# End-to-end pipeline: feature extraction -> notebook (training, evaluation, submission).
# Every step caches its output in features/ and is skipped if that output exists.
# Expects the competition data in Dataset_Final/ (train/, test/, train.csv, test.csv).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p features models
PY=${PY:-python}
run() { local out=$1; shift; [ -e "$out" ] && echo "skip  $out" || { echo "run   $*"; $PY "$@"; }; }

# 1. ASR (resumable). 2 parallel shards: decoding is sequential, so one process
#    under-uses the GPU; more than 2 shards swap on a 16 GB machine.
if [ "$(wc -l < features/transcripts.jsonl 2>/dev/null || echo 0)" -lt 985 ]; then
  $PY src/transcribe.py 0 2 & $PY src/transcribe.py 1 2 & wait
  $PY src/transcribe.py merge
fi
run features/prosody_feats.csv src/prosody_features.py

# 2. Two lanes in parallel: GPU-heavy models / lighter text features
(
  run features/wavlm-base-plus.npz src/audio_embed.py
  run features/wavlm-large.npz src/audio_embed.py microsoft/wavlm-large
  run features/whisper_enc.npz src/whisper_embed.py
  run features/llm_Qwen2.5-7B-Instruct-4bit.csv src/llm_judge.py
  run features/deberta_oof.npy src/finetune_text.py
  for s in 7 2026; do [ -e features/deberta_oof_s$s.npy ] || SEED=$s $PY src/finetune_text.py; done  # extra seeds, averaged
) & gpu=$!
(
  $PY src/text_features.py  # incremental: only clips not yet in the CSV
  $PY src/quick_submit.py || true  # early submission from cheap features (non-fatal)
  run features/text_emb.npz src/text_embed.py
  run features/ling_feats.csv src/linguistic_features.py
) & txt=$!
wait $gpu; wait $txt  # propagate a failure in either lane

# 3. Notebook: trains the stack, reports CV + training RMSE, writes submission.csv
$PY src/build_notebook.py
$PY -m jupyter nbconvert --to notebook --execute --inplace grammar_scoring.ipynb --ExecutePreprocessor.timeout=3600
echo "done -> submission.csv"
