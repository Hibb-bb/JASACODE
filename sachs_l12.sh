source ./.venv/bin/activate

TRAIN_SIZE=100

for seed in 1111 2222 3333 4444 5555; do

    python3 train_sachs.py --train-step 100000 --batch-size 64 --context-len 1000 --seed $seed --num-layers 4 --train-size T$RAIN_SIZE

    python3 eval_sachs_real.py \
      --ckpt-path <CKPT_PATH> \
      --disc-data-dir Sachs/disc_data \
      --output-dir runs/sachs_real_eval_seed$seed \
      --seed $seed
done
