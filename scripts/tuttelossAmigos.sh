# Test Mode B
python ../dvae.py \
  --dataset 'amigos' \
  --backbone 'cbramod' \
  --backbone-weights '/home/user/canWeReally/weights/cbramod_pretrained_weights.pth' \
  --data-file '/home/user/projects/eeg_disentanglement/data/processed_data/amigos_eeg_quadrants_10s_cbramod.pt' \
  --epochs 500 \
  --batch-size 64 \
  --save-dir ../experiments/amigos/amigos_baseline_10s_filt \
  --lr 0.0001 \
  --stage1-epochs 20 \
  --run-name 'amigos_baseline_10s_filt' \
  --use-wandb \