#!/usr/bin/env bash
# Retraining and evaluation of all models (retraining spec, steps 3 and 4).
#
# Each model group is trained and then evaluated directly; main models first, then the ablations:
#   AnEn (evaluation only), A2E, CA2E, SA2E, A2E T3, A2E no_obs (+ AnEn no_obs), A2E euclidean, CA2E euclidean
#
# Usage, in a terminal on the server (from any directory; runs in the repository root):
#   bash experiments/run_retraining.sh             # all steps
#   bash experiments/run_retraining.sh 6           # from step 6 on (e.g. after an abort)
#   bash experiments/run_retraining.sh --list      # list the steps
#
# The output is shown in the terminal and written to logs/retraining/<step>_<module>.log.
# The script stops at the first failed step. Running it again continues the training: models with a history JSON
# (finished runs) are skipped. For a model that was interrupted (model file without history JSON), Api.train asks
# in the terminal: 1 = resume, 2 = start over. Option 2 gives a training run as in the paper; resuming trains
# another 100 epochs with a restarted learning-rate schedule and early-stopping patience.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=".venv/bin/python"
LOG_DIR="logs/retraining"
export PYTHONUNBUFFERED=1

STEPS=(
  # Main models
  "experiments.evaluation.AnEn.anen_metrics"
  "experiments.Trained_Models.A2E.A2E_Train"
  "experiments.evaluation.A2E.a2e_metrics"
  "experiments.Trained_Models.CA2E.CA2E_Train"
  "experiments.evaluation.CA2E.ca2e_metrics"
  "experiments.Trained_Models.SA2E.SA2E_Train"
  "experiments.evaluation.SA2E.sa2e_metrics"
  # Ablations
  "experiments.Trained_Models.A2E.A2E_T3_Train"
  "experiments.evaluation.AblationStudy.a2e_t3_metrics"
  "experiments.Trained_Models.A2E.A2E_no_obs_Train"
  "experiments.evaluation.AblationStudy.a2e_no_obs_metrics"
  "experiments.evaluation.AblationStudy.anen_no_obs_metrics"
  "experiments.Trained_Models.A2E.A2E_euclidean_distance_Train"
  "experiments.evaluation.AblationStudy.a2e_euclidean_distance_metrics"
  "experiments.Trained_Models.CA2E.CA2E_euclidean_distance_Train"
  "experiments.evaluation.AblationStudy.ca2e_euclidean_distance_metrics"
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
echo "$(date '+%F %T')  Retraining in $(pwd), branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "$(date '+%F %T')  Steps $FIRST-${#STEPS[@]}, logs in $LOG_DIR"

for (( i = FIRST - 1; i < ${#STEPS[@]}; i++ )); do
  module="${STEPS[$i]}"
  step=$(printf "%02d" $((i + 1)))
  log="$LOG_DIR/${step}_${module##*.}.log"
  start=$(date +%s)
  echo
  echo "$(date '+%F %T')  ===== [$step/${#STEPS[@]}] $module  (log: $log)"

  # Output to the terminal and the log; input (prompt of Api.train) comes from the terminal
  if ! "$PY" -m "$module" 2>&1 | tee "$log"; then
    echo "$(date '+%F %T')  ===== [$step/${#STEPS[@]}] FAILED after $(( ($(date +%s) - start) / 60 )) min, see $log" >&2
    echo "Continue with: bash experiments/run_retraining.sh $((i + 1))" >&2
    exit 1
  fi
  echo "$(date '+%F %T')  ===== [$step/${#STEPS[@]}] done in $(( ($(date +%s) - start) / 60 )) min"
done

echo
echo "$(date '+%F %T')  All steps done."
