# Test Mode B
uv run ../dvae.py \
  --backbone 'cbramod' \
  --backbone-weights '/mnt/pve/Rita-Storage-2/disentangleData/weights/cbramod_pretrained_weights.pth' \
  --data-file '/mnt/pve/Rita-Storage-2/disentangleData/processed_data/MI_eeg_cbramod.pt' \
  --epochs 100 \
  --batch-size 64 \
  --save-dir ../experiments/MI/test_salvami\
  --lr 0.0001 \
  --stage1-epochs 0 \
  --run-name 'test_salvami' \
  --exclude_tasks 4 \
  --project-name "DREAMER_CBraMod" \
  --classifier_type "diva_classifier" \
  --use-wandb \