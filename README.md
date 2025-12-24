python train.py \
  --batch-size 64 \
  --context-len 200 \
  --graph tree \
  --train-step 200000 \
  --init-lr 1e-4 \
  --train-size 50000 \
  --test-size 10000 \
  --output-dir runs/tree_icl \
  --seed 42
