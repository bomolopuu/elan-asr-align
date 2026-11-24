#!/bin/bash
# SLURM template: part 1 (recording → speech segments → omniASR → .eaf) and,
# optionally, part 2 (put an existing transcription onto the real time of the
# recording and merge it into that .eaf).
#
# Replace every <…> below, then submit:   sbatch slurm/asr_to_eaf.slurm
# To process many recordings, turn AUDIO into a loop or use a job array.
#
#SBATCH --job-name=asr_to_eaf
#SBATCH --partition=<PARTITION>          # a partition with GPUs
#SBATCH --gres=gpu:1                     # some clusters also offer shared GPUs (e.g. --gres=mps:100)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=<PROJECT_DIR>/logs/%x_%j.log

set -euo pipefail

# module load <PYTHON_MODULE>            # if your cluster uses environment modules
source <VENV>/bin/activate               # the environment from requirements-asr.txt

# Model weights are large (several GB for the bigger models). Keep the caches on
# a file system with enough quota instead of $HOME.
export FAIRSEQ2_CACHE_DIR=<LARGE_DISK>/fairseq2_cache   # omniASR weights (part 1)
export TORCH_HOME=<LARGE_DISK>/torch_hub                # MMS_FA weights, ~1.2 GB (part 2)

# Optional: weights already on disk (e.g. offline GPU nodes), see README.
# export FAIRSEQ2_ASSET_DIR=<PROJECT_DIR>/model_cards

# Optional: only if fairseq2n fails with "libffi.so.7" / "libssl.so.1.1" not found.
# export LD_LIBRARY_PATH=<DIR_WITH_THOSE_LIBS>:${LD_LIBRARY_PATH:-}

REPO=<PROJECT_DIR>/<REPO_DIR>
AUDIO=<PROJECT_DIR>/data/<RECORDING>.wav
LANG_CODE=<ISO639-3>_Latn                # e.g. eng_Latn; see README, "Language codes"
MODEL=omniASR_LLM_300M_v2

# Part 1
python "$REPO/asr_to_eaf.py" "$AUDIO" --model "$MODEL" --lang "$LANG_CODE" \
    --batch-size 8 -o "${AUDIO%.*}_asr.eaf"

# Part 2 (optional): a transcription in .eaf (top tier = sentences, with any
# dependent tiers) or .txt (one sentence per line).
# REFERENCE=<PROJECT_DIR>/data/<TRANSCRIPTION>.eaf
# python "$REPO/attach_reference.py" "$AUDIO" "$REFERENCE" \
#     --base "${AUDIO%.*}_asr.eaf" -o "${AUDIO%.*}_asr_ref.eaf"
