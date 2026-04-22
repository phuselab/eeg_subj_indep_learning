# Test Mode B
python ../dvae.py \
  --dataset 'stress_vr' \
  --backbone 'cbramod' \
  --backbone-weights '/home/user/canWeReally/weights/cbramod_pretrained_weights.pth' \
  --data-file '/home/user/projects/eeg_disentanglement/data/processed_data/VR_EEG_Stress_GSR_tonic_equal_thresh_on_minmax_tonic_cbramod.pt' \
  --epochs 500 \
  --batch-size 64 \
  --save-dir ../experiments/STRESS_VR_GSR_tonic_min_max/STRESS_VR_GSR_tonic_min_max_baseline_weighted_class_focal \
  --lr 0.0001 \
  --stage1-epochs 20 \
  --run-name 'STRESS_VR_GSR_tonic_min_max_baseline_weighted_class_focal' \
  --use-wandb \