# Test Mode B
python ../dvae.py \
  --dataset 'dreamer' \
  --backbone 'cbramod' \
  --backbone-weights '/home/user/canWeReally/weights/cbramod_pretrained_weights.pth' \
  --data-file '/home/user/projects/eeg_disentanglement/data/processed_data/dreamer_eeg_quadrants_wo_cross_10s_cbramod.pt' \
  --epochs 500 \
  --batch-size 64 \
  --save-dir ../experiments/DREAMER/DREAMER_baseline_10s_wo_cross \
  --lr 0.0001 \
  --stage1-epochs 20 \
  --run-name 'DREAMER_baseline_10s_wo_cross' \
  --use-wandb \