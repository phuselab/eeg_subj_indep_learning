# Test Mode B
python ../dvae.py \
  --dataset dreamer \
  --backbone 'cbramod' \
  --backbone-weights '/home/user/canWeReally/weights/cbramod_pretrained_weights.pth' \
  --data-file '/home/user/projects/eeg_disentanglement/data/processed_data/dreamer_eeg_quadrants_cbramod.pt' \
  --epochs 1 \
  --batch-size 64 \
  --save-dir ../experiments/testtestdream \
  --lr 0.0001 \
  --stage1-epochs 1 \
  --run-name 'testtestdream' \
  #--use-wandb \