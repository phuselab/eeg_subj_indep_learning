
#!/bin/bash
set -euo pipefail

ENV_PATH="../.env"

# ── Load Environment Variables ───────────────────────────────
if [ -f "$ENV_PATH" ]; then
    echo "✅ Loaded variables from $ENV_PATH"
    # IMPORTANTE: usa "$ENV_PATH" anche dentro il grep!
    export $(grep -v '^#' "$ENV_PATH" | xargs)
else
    echo "⚠️ File $ENV_PATH non trovato, uso i valori di default dello script"
    # fallback
    export DATA_ROOT="${DATA_ROOT:-/mnt/pve/Rita-Storage-2}"
fi

# ── Global Environment ───────────────────────────────────────
PROJECT_NAME="MI_NO_BACKBONE"
BASE_SAVE_DIR="../../../experiments/MI_NO_BACKBONE"
MAIN_SCRIPT="../../../main.py"
DATA_FILE_NAME="MI_eeg_cbramod.pt"

# ── Default Hyperparameters ──────────────────────────────────
# These are used unless overridden in the run_experiment call
DEFAULT_LR="0.0001"
DEFAULT_RECON="1.0"
DEFAULT_KL="0.00001"
DEFAULT_NOISE_KL="0.000001"
DEFAULT_CLASS="0.5"
DEFAULT_VAR_CLASS="1.0"
DEFAULT_INTRA="0.5"
DEFAULT_CROSS_CLASS="0.1"
DEFAULT_CROSS_CYC="0.05"
DEFAULT_SELF_CYC="0.05"
DEFAULT_KD="0.1"
DEFAULT_ADV="0.1"
DEFAULT_CHANNELS="64 32 16 8"

# ── Core Runner Function ─────────────────────────────────────
run_experiment() {
    local run_id="$1"; shift
    local backbone="$1"; shift
    local yaml_path="$1"; shift
    local epochs="$1"; shift
    local stage1_epochs="$1"; shift
    local analysis_block="$1"; shift
    local batch_size="$1" ; shift
    local extra_args="$@"

    local save_path="${BASE_SAVE_DIR}/${run_id}"

    # Definisce il classifier di default
    local classifier_args=("--classifier-type" "${backbone}_classifier")

    # Controlla se --classifier-type è stato passato da fuori
    for arg in "${extra_args[@]}"; do
        if [[ "$arg" == "--classifier-type" ]]; then
            # Trovato! Svuota l'array di default, useremo quello passato negli extra_args
            classifier_args=()
            break
        fi
    done
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  BLOCK/RUN: ${run_id} | BACKBONE: ${backbone}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # We use 'uv run' as requested
    uv run "${MAIN_SCRIPT}" \
      --exclude_tasks 4 \
      --backbone "${backbone}" \
      --backbone-weights "${DATA_ROOT}/disentangleData/weights/${backbone}_pretrained_weights.pth" \
      --data-file "${DATA_ROOT}/disentangleData/processed_data/${DATA_FILE_NAME}" \
      --epochs "${epochs}" \
      --stage1-epochs "${stage1_epochs}" \
      --batch-size "${batch_size}" \
      --save-dir "${save_path}" \
      --lr "${DEFAULT_LR}" \
      --run-name "${run_id}" \
      --project-name "${PROJECT_NAME}" \
      --loss-reconstruction-weight "${DEFAULT_RECON}" \
      --loss-kl-weight "${DEFAULT_KL}" \
      --loss-noise-kl-weight "${DEFAULT_NOISE_KL}" \
      --loss-class-weight "${DEFAULT_CLASS}" \
      --loss-var-class-weight "${DEFAULT_VAR_CLASS}" \
      --loss-intra-weight "${DEFAULT_INTRA}" \
      --loss-cross-class-weight "${DEFAULT_CROSS_CLASS}" \
      --loss-cross-cycle-weight "${DEFAULT_CROSS_CYC}" \
      --loss-self-cycle-weight "${DEFAULT_SELF_CYC}" \
      --loss-kd-weight "${DEFAULT_KD}" \
      --loss-adv-weight "${DEFAULT_ADV}" \
      --mid_channels ${DEFAULT_CHANNELS} \
      --yaml-config "${yaml_path}" \
      --analysis-block "${analysis_block}" \
      --use-wandb \
      "${classifier_args[@]}" \
      ${extra_args} || {
        echo "❌ ATTENZIONE: L'esperimento ${run_id} è andato in errore e si è interrotto!"
        echo "⏭️  Passo automaticamente al prossimo esperimento in coda..."
    }
}

# ── Experiment Groups ────────────────────────────────────────

# Group 1: Comparison of Backbones on Block 1
run_group_1() {
    local config_dir="../../../configs/yamls/block_1"
    local epochs=0
    local stage1_epochs=30
    local analysis_block="lbm"
    local batch_size=128
    
    # Usage: run_experiment [RUN_NAME] [BACKBONE] [YAML_CONFIG] [OPTIONAL_OVERRIDES]
    run_experiment "1A_cbramod" "cbramod" "${config_dir}/A_backbone_frozen.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "${batch_size}"
    run_experiment "1A_labram"   "labram"   "${config_dir}/A_backbone_frozen.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "${batch_size}"
    run_experiment "1A_eegpt"   "eegpt"   "${config_dir}/A_backbone_frozen.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "${batch_size}"
    run_experiment "1B_cbramod" "cbramod" "${config_dir}/B_backbone_trainable.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "${batch_size}"
    run_experiment "1B_labram"   "labram"   "${config_dir}/B_backbone_trainable.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "${batch_size}"
    run_experiment "1B_eegpt"   "eegpt"   "${config_dir}/B_backbone_trainable.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "${batch_size}"
}

# Group 2: Specific Task/Subject variations
run_group_2() {
    # Definizione delle variabili comuni locali
    local model="cbramod"
    local config_dir="../../../configs/yamls/block_2"
    local epochs=200
    local stage1_epochs=0
    local analysis_block="disentanglement"
    local batch_size=128
    local classifier_type="--classifier-type diva_classifier"

    #run_experiment "2A" "${model}" "${config_dir}/A_task_only.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "${batch_size}" "${classifier_type}"
    #run_experiment "2B" "${model}" "${config_dir}/B_task_subj.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "${batch_size}" "${classifier_type}"
    #run_experiment "2C" "${model}" "${config_dir}/C_task_subj_noise_variational.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "${batch_size}" "${classifier_type}"
    run_experiment "2D" "${model}" "${config_dir}/D_generator_reconstruction.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "${batch_size}" "${classifier_type}"
    #run_experiment "2E" "${model}" "${config_dir}/E_discriminator_full_DIVA.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "64" "${classifier_type}"
    #run_experiment "2F" "${model}" "${config_dir}/F_DIVA_no_UNET.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "64" "${classifier_type}"
    # run_experiment "2G" "${model}" "${config_dir}/G_generator_reconstruction_no_UNET.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "64" "${classifier_type}"
}

    
    
run_group_3() {
    local model="cbramod"
    local config_dir="../../../configs/yamls/block_3"
    local epochs=0
    local stage1_epochs=30
    local analysis_block="alignment"
    local batch_size=128

    run_experiment "3A" "${model}" "${config_dir}/A_Euclidean_Alignment.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "${batch_size}"
    run_experiment "3B" "${model}" "${config_dir}/B_Latent_Alignment2D.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "${batch_size}"
    run_experiment "3C" "${model}" "${config_dir}/C_AdaptiveBatchNorm.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "${batch_size}"
    run_experiment "3D" "${model}" "${config_dir}/D_BatchNorm.yaml" "${epochs}" "${stage1_epochs}" "${analysis_block}" "${batch_size}"

}

# ── Dispatcher ───────────────────────────────────────────────
case "${1:-all}" in
    "1")   run_group_1 ;;
    "2")   run_group_2 ;;
    "3")   run_group_3 ;;
    "all") run_group_1; run_group_2; run_group_3 ;;
    "12")  run_group_1; run_group_3 ;;
    "13")  run_group_1; run_group_3 ;;
    "23")  run_group_2; run_group_3 ;;
    *)     echo "Usage: $0 {1|2|3|12|13|23|all}"; exit 1 ;;
esac