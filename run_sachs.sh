source ./.venv/bin/activate

for seed in 1111 2222 3333 4444 5555; do

    python3 train_sachs.py --train-step 100000 --batch-size 64 --context-len 1000 --seed 1111 --num-layers 4 --train-size 100

    python3 eval_sachs_real.py \
      --ckpt-path "/projects/b1094/ywl7940/JASACODE/outputs/sachs/seed_${seed}/1000/L2/100/logs/version_0/checkpoints/best.ckpt" \
      --disc-data-dir Sachs/disc_data \
      --output-dir runs/sachs_real_eval_seed$seed \
      --seed $seed

done

mkdir -p runs/sachs_real_eval_qp/seed_{1111,2222,3333,4444,5555}


cp runs/sachs_real_eval_seed1111/*_eval_tv_real.csv runs/sachs_real_eval_qp/seed_1111/
cp runs/sachs_real_eval_seed2222/*_eval_tv_real.csv runs/sachs_real_eval_qp/seed_2222/
cp runs/sachs_real_eval_seed3333/*_eval_tv_real.csv runs/sachs_real_eval_qp/seed_3333/
cp runs/sachs_real_eval_seed4444/*_eval_tv_real.csv runs/sachs_real_eval_qp/seed_4444/
cp runs/sachs_real_eval_seed5555/*_eval_tv_real.csv runs/sachs_real_eval_qp/seed_5555/


uv run python quick_plot_sachs.py \
  --base-dir ./runs/sachs_real_eval_qp \
  --seeds 1111 2222 3333 4444 5555 \
  --out-dir ./imgs/sachs_real_avg5
