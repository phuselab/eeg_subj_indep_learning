 # Test Mode B
python ../dvae.py \
  --dataset MI_eeg \
  --backbone 'cbramod' \
  --backbone-weights '/home/user/canWeReally/weights/cbramod_pretrained_weights.pth' \
  --data-file '/home/user/canWeReally/data/processed_data/MI_eeg_cbramod.pt' \
  --epochs 1500 \
  --batch-size 64 \
  --save-dir ../experiments/allLossesNewEnc \
  --lr 0.0001 \
  --stage1-epochs 10 \
  --run-name 'allLossesNewEnc' \
  --use-wandb \