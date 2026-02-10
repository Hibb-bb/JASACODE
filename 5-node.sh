for seed in 1111 2222 3333 4444 5555; do

  python train.py \
    --batch-size 16 \
    --min-context-len 100 \
    --max-context-len 500 \
    --graph tree5 \
    --train-step 20000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --warmup-steps 2000 \
    --min-lr 1e-6 \
    --init-lr 3e-4 \
    --seed $seed


  python train.py \
    --batch-size 16 \
    --min-context-len 100 \
    --max-context-len 500 \
    --graph chain5 \
    --train-step 20000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --warmup-steps 2000 \
    --min-lr 1e-6 \
    --init-lr 3e-4 \
    --seed $seed


  python train.py \
    --batch-size 16 \
    --min-context-len 100 \
    --max-context-len 500 \
    --graph general5 \
    --train-step 20000 \
    --init-lr 1e-4 \
    --train-size 20000 \
    --test-size 5000 \
    --output-dir runs/ \
    --warmup-steps 2000 \
    --min-lr 1e-6 \
    --init-lr 3e-4 \
    --seed $seed

done

python3 quick_plot.py --graph tree5
python3 quick_plot.py --graph general5
python3 quick_plot.py --graph chain5