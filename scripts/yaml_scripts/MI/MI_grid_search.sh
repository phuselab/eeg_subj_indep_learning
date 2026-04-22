#!/bin/bash
# ============================================================
# Grid search — Experiment E loss weights  (≤ 10 runs)
#
# Architecture of the search:
#
#   Run  1        — anchor (all defaults, E config)
#   Runs 2–4      — GROUP C: cls vs var_cls ratio at fixed sum ~1.5
#                   (which supervision signal matters more?)
#   Runs 5–6      — GROUP D: self-reconstruction weight
#                   (how hard should the generator be pushed?)
#   Runs 7–9      — GROUP E: intra_class vs cross_class contrastive
#                   (same-class cohesion vs diff-class separation)
#   Run  10       — JOINT: best values from groups C/D/E combined
#
# Usage:
#   chmod +x grid_search_E.sh
#   ./grid_search_E.sh              # run all 10
#   ./grid_search_E.sh --dry-run    # print commands only
# ============================================================

set -euo pipefail

# ── Load .env ────────────────────────────────────────────────
ENV_PATH="../.env"
if [ -f "$ENV_PATH" ]; then
    echo "✅ Loaded variables from $ENV_PATH"
    export $(grep -v '^#' "$ENV_PATH" | xargs)
else
    echo "⚠️  $ENV_PATH not found — using defaults"
    export DATA_ROOT="${DATA_ROOT:-/mnt/pve/Rita-Storage-2}"
fi

# ── Global config ─────────────────────────────────────────────
PROJECT_NAME="CLARE_dove_sei"
BASE_SAVE_DIR="../../../experiments/CLARE/grid_E"
MAIN_SCRIPT="../../../main.py"
DATA_FILE_NAME="clare_processed_thresh_5_cbramod_18Parts_2channels.pt"

MODEL="cbramod"
YAML_CONFIG="../../../configs/yamls/block_2/E_discriminator_full_DIVA.yaml"
EPOCHS=150
STAGE1_EPOCHS=0
BATCH_SIZE=256
ANALYSIS_BLOCK="disentanglement"
CLS_ARGS="--classifier-type diva_classifier"
DEFAULT_CHANNELS="64 32 16 8"
LR="0.0001"

# ── Anchor (default) weights ──────────────────────────────────
A_RECON="0.5"        # self_reconstruction_weight
A_KL="0.00001"
A_NKL="0.000001"
A_CLS="0.5"          # classification_weight
A_VCLS="1.0"         # var_classification_weight
A_INTRA="0.5"        # cross_subject_intra_class_weight
A_CC="0.1"           # cross_subject_cross_class_weight
A_SCYC="0.05"        # self_cycle_weight
A_XCYC="0.05"        # cross_cross_cycle_weight
A_KD="0.1"
A_ADV="0.1"

# ── Parse CLI ─────────────────────────────────────────────────
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Runner ────────────────────────────────────────────────────
run_experiment() {
    local run_id="$1"
    local recon="$2"
    local cls="$3"
    local vcls="$4"
    local intra="$5"
    local cc="$6"

    local save_path="${BASE_SAVE_DIR}/${run_id}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  RUN: ${run_id}"
    printf "  recon=%-6s  cls=%-5s  vcls=%-5s  intra=%-5s  cc=%s\n" \
           "$recon" "$cls" "$vcls" "$intra" "$cc"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local cmd="uv run ${MAIN_SCRIPT} \
        --backbone ${MODEL} \
        --backbone-weights /home/user/projects/eeg_disentanglement/data/weights/${MODEL}_pretrained_weights.pth \
        --data-file /home/user/projects/eeg_disentanglement/data/processed_data/${DATA_FILE_NAME} \
        --epochs ${EPOCHS} \
        --stage1-epochs ${STAGE1_EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --save-dir ${save_path} \
        --lr ${LR} \
        --run-name ${run_id} \
        --project-name ${PROJECT_NAME} \
        --loss-reconstruction-weight ${recon} \
        --loss-kl-weight ${A_KL} \
        --loss-noise-kl-weight ${A_NKL} \
        --loss-class-weight ${cls} \
        --loss-var-class-weight ${vcls} \
        --loss-intra-weight ${intra} \
        --loss-cross-class-weight ${cc} \
        --loss-cross-cycle-weight ${A_XCYC} \
        --loss-self-cycle-weight ${A_SCYC} \
        --loss-kd-weight ${A_KD} \
        --loss-adv-weight ${A_ADV} \
        --mid_channels ${DEFAULT_CHANNELS} \
        --yaml-config ${YAML_CONFIG} \
        --analysis-block ${ANALYSIS_BLOCK} \
        --use-wandb \
        ${CLS_ARGS}"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $cmd"
        return
    fi

    mkdir -p "${save_path}"
    eval "$cmd" || {
        echo "❌ Run ${run_id} failed — skipping to next."
    }
}

# ─────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         Experiment E — Loss Weight Grid Search           ║"
echo "║  1 anchor | 3 cls | 2 recon | 3 contrastive | 1 joint   ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Run 1: Anchor ─────────────────────────────────────────────
# All defaults. This is the baseline every other run is compared to.
run_experiment \
    "E_01_anchor" \
    "$A_RECON" "$A_CLS" "$A_VCLS" "$A_INTRA" "$A_CC"

# ── Runs 2–4: GROUP C — cls vs var_cls ratio ──────────────────
# Total classification pressure fixed at 1.5.
# Sweeps how the supervision budget is split between the two heads.
# Reconstruction and contrastive terms held at anchor values.
#
#   run  cls   vcls   (sum = 1.5)
#    2   0.25  1.25   — lean heavily on var_cls (latent-space supervision)
#    3   0.75  0.75   — balanced
#    4   1.25  0.25   — lean heavily on cls (log-space supervision)
run_experiment "E_02_cls_light"    "$A_RECON" "0.25" "1.25" "$A_INTRA" "$A_CC"
run_experiment "E_03_cls_balanced" "$A_RECON" "0.75" "0.75" "$A_INTRA" "$A_CC"
run_experiment "E_04_cls_heavy"    "$A_RECON" "1.25" "0.25" "$A_INTRA" "$A_CC"

# ── Runs 5–6: GROUP D — self-reconstruction weight ────────────
# Classification and contrastive held at anchor values.
# Two levels only: weaker (let the classifier dominate) vs
# stronger (push the generator harder).
#
#   run  recon
#    5   0.1   — light reconstruction pressure
#    6   1.0   — strong reconstruction pressure
run_experiment "E_05_recon_light"  "0.1"  "$A_CLS" "$A_VCLS" "$A_INTRA" "$A_CC"
run_experiment "E_06_recon_strong" "1.0"  "$A_CLS" "$A_VCLS" "$A_INTRA" "$A_CC"

# ── Runs 7–9: GROUP E — intra_class vs cross_class ────────────
# The push/pull pair specific to the full E configuration.
# Reconstruction and classification held at anchor values.
#
#   run  intra  cc    note
#    7   0.5    0.5   — equal pull and push (more separation pressure)
#    8   1.0    0.1   — strong cohesion, weak separation (default-ish)
#    9   1.0    0.5   — strong cohesion + more separation
run_experiment "E_07_contrastive_balanced" "$A_RECON" "$A_CLS" "$A_VCLS" "0.5" "0.5"
run_experiment "E_08_contrastive_cohesion" "$A_RECON" "$A_CLS" "$A_VCLS" "1.0" "0.1"
run_experiment "E_09_contrastive_both"     "$A_RECON" "$A_CLS" "$A_VCLS" "1.0" "0.5"

# ── Run 10: JOINT — combine best from each group ─────────────
# !! Update these with the winners from runs 2–9 before launching !!
# Defaults below match the anchor so this run is safe to execute
# even before reviewing results — it will simply replicate the anchor.
BEST_RECON="0.5"    # winner from group D (runs 5–6)
BEST_CLS="0.75"     # winner from group C (runs 2–4)
BEST_VCLS="0.75"    # paired with BEST_CLS
BEST_INTRA="1.0"    # winner from group E (runs 7–9)
BEST_CC="0.1"       # paired with BEST_INTRA

run_experiment \
    "E_10_joint" \
    "$BEST_RECON" "$BEST_CLS" "$BEST_VCLS" "$BEST_INTRA" "$BEST_CC"

echo ""
echo "✓ All 10 runs complete."