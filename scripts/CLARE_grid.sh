#!/bin/bash
set -euo pipefail

# ── Base Config (Update paths as needed) ──────────────────────
BACKBONE="cbramod"
BACKBONE_WEIGHTS="/mnt/pve/Rita-Storage-2/disentangleData/weights/cbramod_pretrained_weights.pth"
DATA_FILE="/mnt/pve/Rita-Storage-2/disentangleData/processed_data/clare_processed_cbramod.pt"
PROJECT_NAME="CBraMod_CLARE_Grid_V2"
BASE_SAVE_DIR="../experiments/CLARE/grid_v2"
DVAE_SCRIPT="../dvae.py"
MID_CHANNELS=(4 4)  # Default mid_channels for all runs (can be overridden)
STAGE1_EPOCHS=0
EPOCHS=200
BATCH_SIZE=1024
LR=0.0001


# ── CLI Parsing ──────────────────────────────────────────────
RUN_GROUP="all"
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --group)   RUN_GROUP="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Anchor Weights (From your LossConfig defaults) ───────────
# Note: These acts as the "control" variables.
DEF_REC="1.0"
DEF_KL="0.00001"
DEF_N_KL="0.000001"
DEF_CLS="0.5"
DEF_V_CLS="1.0"
DEF_S_CYC="0.05"
DEF_INTRA="0.5"
DEF_CROSS="0.1"
DEF_C_CYC="0.05"
DEF_KD="0.1"
DEF_ADV="0.1"

run_exp() {
    local name="$1"; shift
    local args="$@"
    local sdir="${BASE_SAVE_DIR}/${name}"
    local cmd=(
        uv run "${DVAE_SCRIPT}"
        --backbone "${BACKBONE}"
        --backbone-weights "${BACKBONE_WEIGHTS}"
        --data-file "${DATA_FILE}"
        --save-dir "${sdir}"
        --run-name "${name}"
        --use-wandb
        --mid_channels "${MID_CHANNELS[@]}"
        --epochs "${EPOCHS}"
        --batch-size "${BATCH_SIZE}"
        --lr "${LR}"
        --stage1-epochs "${STAGE1_EPOCHS}"
        --project-name "${PROJECT_NAME}"
        "${extra_args[@]}"
    )

    echo -e "\n>> Running: ${name}"
    if [ "$DRY_RUN" = true ]; then 
        echo "[DRY] ${cmd[*]}"
    else 
        mkdir -p "$sdir"
        "${cmd[@]}" # Execute the array directly (no eval needed!)
    fi
}

# ── GROUP 1: The VAE Compression Trade-off ────────────────────
# Relationship between Reconstruction and KL (Beta-VAE logic)
run_group_1() {
    echo "=== G1: Recon vs KL ==="
    for rec in 0.1 1.0 5.0; do
        for kl in 1e-6 1e-5 1e-4; do
            # Pass the flags as separate items
            run_exp "g1_rec${rec}_kl${kl}" --loss-reconstruction-weight "${rec}" --loss-kl-weight "${kl}"
        done
    done
}

# ── GROUP 2: Supervision Depth ────────────────────────────────
# Does the model benefit from logit-level (cls) or latent-level (var_cls) supervision?
run_group_2() {
    echo "=== G2: Logit vs Latent Supervision (5 runs) ==="
    # Constant sum of 1.5, shifting the weight
    local pairs="0.0,1.5 0.5,1.0 0.75,0.75 1.0,0.5 1.5,0.0"
    for p in $pairs; do
        run_exp "g2_cls${p%,*}_vcls${p#*,}" \
            --loss-class-weight ${p%,*} --loss-var-class-weight ${p#*,}
    done
}

# ── GROUP 3: Geometric Constraints ────────────────────────────
# Pulling same-subject (Intra) vs Pushing different-subject (Cross)
run_group_3() {
    echo "=== G3: Latent Geometry (9 runs) ==="
    for intra in 0.1 0.5 1.0; do
        for cross in 0.01 0.1 0.3; do
            run_exp "g3_intra${intra}_cross${cross}" \
                --loss-intra-weight ${intra} --loss-cross-class-weight ${cross}
        done
    done
}

# ── GROUP 4: Cycle Consistency ────────────────────────────────
# Testing Self-Cycle vs Cross-Class Cycle
run_group_4() {
    echo "=== G4: Cycle Consistency (4 runs) ==="
    for scyc in 0.01 0.1; do
        for ccyc in 0.01 0.1; do
            run_exp "g4_scyc${scyc}_ccyc${ccyc}" \
                --loss-self-cycle-weight ${scyc} --loss-cross-cycle-weight ${ccyc}
        done
    done
}

# ── GROUP 5: Regularization Balance ───────────────────────────
# Knowledge Distillation (Stability) vs Adversarial (Disentanglement)
run_group_5() {
    echo "=== G5: Regularizers (4 runs) ==="
    for kd in 0.1 0.5; do
        for adv in 0.01 0.1; do
            run_exp "g5_kd${kd}_adv${adv}" \
                --loss-kd-weight ${kd} --loss-adv-weight ${adv}
        done
    done
}

# ── Dispatcher ───────────────────────────────────────────────
case "$RUN_GROUP" in
    1) run_group_1 ;; 2) run_group_2 ;; 3) run_group_3 ;;
    4) run_group_4 ;; 5) run_group_5 ;;
    all) run_group_1; run_group_2; run_group_3; run_group_4; run_group_5 ;;
    *) echo "Invalid group"; exit 1 ;;
esac