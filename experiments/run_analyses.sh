#!/usr/bin/env bash
# Analyses after the retraining (run_retraining.sh must be finished):
#   AnEn grid of predictor weights and k (appendix), inference modes and k, significance of the score differences,
#   PIT histograms, embeddings and analogs
# Afterwards run the notebooks evaluation/baselines/baselines.ipynb and evaluation/AnEn/anen_optimization.ipynb.
#
# Usage, in a terminal on the server (from any directory; runs in the repository root):
#   bash experiments/run_analyses.sh             # all steps
#   bash experiments/run_analyses.sh 4           # from step 4 on (e.g. after an abort)
#   bash experiments/run_analyses.sh --list      # list the steps
#
# The output is shown in the terminal and written to logs/analyses/<step>_<module>.log.
# The script stops at the first failed step.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=".venv/bin/python"
LOG_DIR="logs/analyses"
export PYTHONUNBUFFERED=1

# Module and arguments
STEPS=(
  "experiments.evaluation.AnEn.anen_optimization"
  "experiments.evaluation.InferenceModes.inference_modes_metrics A2E"
  "experiments.evaluation.InferenceModes.inference_modes_metrics CA2E"
  "experiments.evaluation.InferenceModes.inference_modes_metrics SA2E"
  "experiments.evaluation.InferenceModes.inference_modes_report"
  "experiments.evaluation.Significance.significance"
  "experiments.evaluation.pit_histogram"
  "experiments.evaluation.Embeddings.analog_comparison A2E"
  "experiments.evaluation.Embeddings.embedding_projection A2E 0"
)

if [[ "${1:-}" == "--list" ]]; then
  for i in "${!STEPS[@]}"; do printf "%2d  %s\n" $((i + 1)) "${STEPS[$i]}"; done
  exit 0
fi

FIRST="${1:-1}"
if ! [[ "$FIRST" =~ ^[0-9]+$ ]] || (( FIRST < 1 || FIRST > ${#STEPS[@]} )); then
  echo "First step must be a number between 1 and ${#STEPS[@]} (see --list)." >&2
  exit 1
fi

if [[ ! -x "$PY" ]]; then
  echo "No Python environment at $PY (repository root: $(pwd))." >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
echo "$(date '+%F %T')  Analyses in $(pwd), branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "$(date '+%F %T')  Steps $FIRST-${#STEPS[@]}, logs in $LOG_DIR"

for (( i = FIRST - 1; i < ${#STEPS[@]}; i++ )); do
  read -r -a cmd <<< "${STEPS[$i]}"
  module="${cmd[0]}"
  step=$(printf "%02d" $((i + 1)))
  suffix=$(IFS=_; echo "${cmd[*]:1}")
  log="$LOG_DIR/${step}_${module##*.}${suffix:+_$suffix}.log"
  start=$(date +%s)
  echo
  echo "$(date '+%F %T')  ===== [$step/${#STEPS[@]}] ${STEPS[$i]}  (log: $log)"

  if ! "$PY" -m "${cmd[@]}" 2>&1 | tee "$log"; then
    echo "$(date '+%F %T')  ===== [$step/${#STEPS[@]}] FAILED after $(( ($(date +%s) - start) / 60 )) min, see $log" >&2
    echo "Continue with: bash experiments/run_analyses.sh $((i + 1))" >&2
    exit 1
  fi
  echo "$(date '+%F %T')  ===== [$step/${#STEPS[@]}] done in $(( ($(date +%s) - start) / 60 )) min"
done

echo
echo "$(date '+%F %T')  All steps done."
