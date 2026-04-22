# Test Mode B
python ../dvae.py \
  --backbone 'cbramod' \
  --backbone-weights '/home/user/canWeReally/weights/cbramod_pretrained_weights.pth' \
  --data-file '/home/user/projects/eeg_disentanglement/data/processed_data/MI_eeg_cbramod.pt' \
  --epochs 200 \
  --batch-size 32 \
  --save-dir ../experiments/MI/MI_cross_cross_cycle \
  --lr 0.0001 \
  --stage1-epochs 0 \
  --run-name 'MI_cross_cross_cycle' \
  --exclude_tasks 4 \
  --use-wandb \
   --project-name "CBraMod_finetune" \