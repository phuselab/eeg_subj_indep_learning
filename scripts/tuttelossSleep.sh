# Test Mode B
python ../dvae.py \
  --dataset sleepedfx \
  --backbone 'cbramod' \
  --backbone-weights '/home/user/canWeReally/weights/cbramod_pretrained_weights.pth' \
  --data-file '/home/user/projects/eeg_disentanglement/data/processed_data/sleepedfx_cbramod_data.pt' \
  --epochs 500 \
  --batch-size 128 \
  --save-dir ../experiments/sleep/SLEEP_baseline \
  --lr 0.0001 \
  --stage1-epochs 20 \
  --run-name 'SLEEP_baseline' \
  --use-wandb \