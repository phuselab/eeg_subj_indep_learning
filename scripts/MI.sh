#!/bin/bash
# ============================================================
# Non-exhaustive grid search over DVAE Stage 2 loss weights
#
# Group structure:
#   G1  ( 3 runs) — KL weights, log-scale
#   G2a ( 5 runs) — cls vs var_cls ratio at fixed sum
#                   (which VAE stage benefits more from supervision)
#   G2b ( 9 runs) — total classification pressure vs latent_consistency
#                   (disentanglement strength vs representation cohesion)
#   G3  ( 9 runs) — cross_subject_intra vs cross_subject_cross
#                   (pull same-class together vs push diff-class apart)
#   G4  ( 9 runs) — kd vs adversarial regularizers
#   G5  ( 4 runs) — joint refinement with best values from G1-G4
#
# Total: 3 + 5 + 9 + 9 + 9 + 4 = 39 runs
#
# Usage:
#   chmod +x MI.sh
#   ./MI.sh                # run all groups sequentially
#   ./MI.sh --group 2a     # run only group 2a
#   ./MI.sh --dry-run      # print commands without running
#
# Parallelism tip (one group per GPU):
#   CUDA_VISIBLE_DEVICES=0 ./MI.sh --group 1   &
#   CUDA_VISIBLE_DEVICES=1 ./MI.sh --group 2a  &
#   CUDA_VISIBLE_DEVICES=2 ./MI.sh --group 2b  &
#   CUDA_VISIBLE_DEVICES=3 ./MI.sh --group 3   &
# ============================================================

set -euo pipefail

# ── Base config ──────────────────────────────────────────────
BACKBONE="cbramod"
BACKBONE_WEIGHTS="/home/user/canWeReally/weights/cbramod_pretrained_weights.pth"
DATA_FILE="/home/user/projects/eeg_disentanglement/data/processed_data/MI_eeg_cbramod.pt"
EPOCHS=100
BATCH_SIZE=64
LR=0.0001
STAGE1_EPOCHS=0
EXCLUDE_TASKS=4
PROJECT_NAME="CBraMod_finetune_MI"
BASE_SAVE_DIR="../experiments/MI/grid_search"
DVAE_SCRIPT="../dvae.py"

# ── Parse CLI args ────────────────────────────────────────────
RUN_GROUP="all"
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --group)   RUN_GROUP="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Default (anchor) weights ──────────────────────────────────
DEFAULT_KL="0.0001"
DEFAULT_NOISE_KL="0.0001"
DEFAULT_CLASS="0.5"
DEFAULT_VAR_CLASS="1.0"
DEFAULT_LATENT_CONS="0.5"
DEFAULT_INTRA="1.0"
DEFAULT_CROSS_CLASS="0.1"
DEFAULT_KD="0.1"
DEFAULT_ADV="0.1"

# ── Runner ────────────────────────────────────────────────────
run_experiment() {
    local run_name="$1"; shift
    local extra_args="$@"

    local save_dir="${BASE_SAVE_DIR}/${run_name}"
    local cmd="python ${DVAE_SCRIPT} \
        --backbone '${BACKBONE}' \
        --backbone-weights '${BACKBONE_WEIGHTS}' \
        --data-file '${DATA_FILE}' \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --save-dir ${save_dir} \
        --project-name ${PROJECT_NAME} \
        --lr ${LR} \
        --stage1-epochs ${STAGE1_EPOCHS} \
        --run-name '${run_name}' \
        --exclude_tasks ${EXCLUDE_TASKS} \
        --use-wandb \
        ${extra_args}"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  RUN: ${run_name}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $cmd"
    else
        mkdir -p "${save_dir}"
        eval "$cmd"
    fi
}

# ─────────────────────────────────────────────────────────────
# GROUP 1 — KL weights (3 runs)
# Both kl and noise_kl swept together (same scale expected).
# ─────────────────────────────────────────────────────────────
run_group_1() {
    echo "=== GROUP 1: KL weights (3 runs) ==="
    for kl in 1e-5 1e-4 1e-3; do
        run_experiment "g1_kl${kl}" \
            --loss-kl-weight ${kl} \
            --loss-noise-kl-weight ${kl} \
            --loss-class-weight ${DEFAULT_CLASS} \
            --loss-var-class-weight ${DEFAULT_VAR_CLASS} \
            --loss-latent-cons-weight ${DEFAULT_LATENT_CONS} \
            --loss-intra-weight ${DEFAULT_INTRA} \
            --loss-cross-class-weight ${DEFAULT_CROSS_CLASS} \
            --loss-kd-weight ${DEFAULT_KD} \
            --loss-adv-weight ${DEFAULT_ADV}
    done
}

# ─────────────────────────────────────────────────────────────
# GROUP 2a — cls vs var_cls ratio at fixed sum (5 runs)
#
# Total classification pressure held constant at ~1.5.
# Varying split between log-space (classification) and
# latent-space (var_classification) supervision.
# Question: does the model benefit more from supervising
# before or after the reparametrization?
# ─────────────────────────────────────────────────────────────
run_group_2a() {
    echo "=== GROUP 2a: cls vs var_cls ratio at fixed sum ~1.5 (5 runs) ==="
    # pairs: (class_w, var_class_w)
    local pairs="0.25,1.25 0.5,1.0 0.75,0.75 1.0,0.5 1.25,0.25"
    for pair in $pairs; do
        local class_w="${pair%,*}"
        local var_class_w="${pair#*,}"
        run_experiment "g2a_cls${class_w}_vcls${var_class_w}" \
            --loss-kl-weight ${DEFAULT_KL} \
            --loss-noise-kl-weight ${DEFAULT_NOISE_KL} \
            --loss-class-weight ${class_w} \
            --loss-var-class-weight ${var_class_w} \
            --loss-intra-weight ${DEFAULT_INTRA} \
            --loss-cross-class-weight ${DEFAULT_CROSS_CLASS} \
            --loss-kd-weight ${DEFAULT_KD} \
            --loss-adv-weight ${DEFAULT_ADV}
    done
}

# ─────────────────────────────────────────────────────────────
# GROUP 2b — total classification pressure vs latent_consistency (9 runs)
#
# Three total pressure levels (low/default/high) crossed with
# three latent_consistency values.
# This is the core tension: how hard to supervise disentanglement
# vs how strongly to enforce subject representation cohesion.
# cls:var_cls ratio held at default 1:2.
# ─────────────────────────────────────────────────────────────
run_group_2b() {
    echo "=== GROUP 2b: total cls pressure vs latent_consistency (9 runs) ==="
    # (class_w, var_class_w): low=0.75 total, default=1.5, high=3.0
    local pressure_pairs="0.25,0.5 0.5,1.0 1.0,2.0"
    for lc in 0.1 0.5 1.0; do
        for pair in $pressure_pairs; do
            local class_w="${pair%,*}"
            local var_class_w="${pair#*,}"
            run_experiment "g2b_cls${class_w}_vcls${var_class_w}_lc${lc}" \
                --loss-kl-weight ${DEFAULT_KL} \
                --loss-noise-kl-weight ${DEFAULT_NOISE_KL} \
                --loss-class-weight ${class_w} \
                --loss-var-class-weight ${var_class_w} \
                --loss-latent-cons-weight ${lc} \
                --loss-intra-weight ${DEFAULT_INTRA} \
                --loss-cross-class-weight ${DEFAULT_CROSS_CLASS} \
                --loss-kd-weight ${DEFAULT_KD} \
                --loss-adv-weight ${DEFAULT_ADV}
        done
    done
}

# ─────────────────────────────────────────────────────────────
# GROUP 3 — intra_class vs cross_class (9 runs)
#
# Genuine push/pull pair:
#   intra_class  — pulls same-class subjects together (cohesion)
#   cross_class  — pushes diff-class subjects apart   (separation)
# Joint sweep reveals where the balance should sit.
# ─────────────────────────────────────────────────────────────
run_group_3() {
    echo "=== GROUP 3: cross_subject_intra vs cross_subject_cross (9 runs) ==="
    for intra in 0.5 1.0 2.0; do
        for cross_class in 0.05 0.1 0.5; do
            run_experiment "g3_intra${intra}_cc${cross_class}" \
                --loss-kl-weight ${DEFAULT_KL} \
                --loss-noise-kl-weight ${DEFAULT_NOISE_KL} \
                --loss-class-weight ${DEFAULT_CLASS} \
                --loss-var-class-weight ${DEFAULT_VAR_CLASS} \
                --loss-intra-weight ${intra} \
                --loss-cross-class-weight ${cross_class} \
                --loss-kd-weight ${DEFAULT_KD} \
                --loss-adv-weight ${DEFAULT_ADV}
        done
    done
}

# ─────────────────────────────────────────────────────────────
# GROUP 4 — Regularizers: kd vs adversarial (9 runs)
#
# Both are auxiliary losses that don't directly drive
# disentanglement. Swept jointly to find a stable range
# that doesn't overpower the main objectives.
# ─────────────────────────────────────────────────────────────
run_group_4() {
    echo "=== GROUP 4: kd vs adversarial regularizers (9 runs) ==="
    for kd in 0.01 0.1 0.5; do
        for adv in 0.01 0.1 0.5; do
            run_experiment "g4_kd${kd}_adv${adv}" \
                --loss-kl-weight ${DEFAULT_KL} \
                --loss-noise-kl-weight ${DEFAULT_NOISE_KL} \
                --loss-class-weight ${DEFAULT_CLASS} \
                --loss-var-class-weight ${DEFAULT_VAR_CLASS} \
                --loss-intra-weight ${DEFAULT_INTRA} \
                --loss-cross-class-weight ${DEFAULT_CROSS_CLASS} \
                --loss-kd-weight ${kd} \
                --loss-adv-weight ${adv}
        done
    done
}

# ─────────────────────────────────────────────────────────────
# GROUP 5 — Joint refinement (4 runs)
#
# Update BEST_* values after reviewing G1-G4 in wandb,
# then run this group to validate the combined configuration.
# ─────────────────────────────────────────────────────────────
run_group_5() {
    echo "=== GROUP 5: joint refinement — update BEST_* before running (4 runs) ==="

    # !! Replace these with the best values found in G1-G4 !!
    BEST_KL="1e-4"
    BEST_CLASS="0.5"
    BEST_VAR_CLASS="1.0"
    BEST_LC="0.5"
    BEST_INTRA="1.0"
    BEST_CC="0.1"
    BEST_KD="0.1"
    BEST_ADV="0.01"

    # Perturb the two most impactful axes found (G2b and G3)
    for lc in ${BEST_LC} 1.0; do
        for intra in ${BEST_INTRA} 2.0; do
            run_experiment "g5_lc${lc}_intra${intra}" \
                --loss-kl-weight ${BEST_KL} \
                --loss-noise-kl-weight ${BEST_KL} \
                --loss-class-weight ${BEST_CLASS} \
                --loss-var-class-weight ${BEST_VAR_CLASS} \
                --loss-latent-cons-weight ${lc} \
                --loss-intra-weight ${intra} \
                --loss-cross-class-weight ${BEST_CC} \
                --loss-kd-weight ${BEST_KD} \
                --loss-adv-weight ${BEST_ADV}
        done
    done
}

# ─────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║            DVAE Loss Weight Grid Search                  ║"
echo "║  G1:3 | G2a:5 | G2b:9 | G3:9 | G4:9 | G5:4 = 39 runs  ║"
echo "╚══════════════════════════════════════════════════════════╝"

case "$RUN_GROUP" in
    1)   run_group_1 ;;
    2a)  run_group_2a ;;
    2b)  run_group_2b ;;
    3)   run_group_3 ;;
    4)   run_group_4 ;;
    5)   run_group_5 ;;
    all)
        run_group_1
        run_group_2a
        run_group_2b
        run_group_3
        run_group_4
        # run_group_5  # uncomment after reviewing G1-G4 results
        ;;
    *)
        echo "Unknown group: ${RUN_GROUP}"
        echo "Valid: 1, 2a, 2b, 3, 4, 5, all"
        exit 1
        ;;
esac

echo ""
echo "✓ Done (group: ${RUN_GROUP})."