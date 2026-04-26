for graph in "chain5" "chain" "tree5" "tree" "general" "general7"; do

    for seed in 1111 2222 3333 4444 5555; do

    uv run python3 train.py \
        --batch-size 64 \
        --context-len 200 \
        --graph $graph \
        --train-step 100000 \
        --init-lr 1e-4 \
        --train-size 20000 \
        --test-size 5000 \
        --output-dir runs/ \
        --warmup-steps 2000 \
        --min-lr 1e-6 \
        --init-lr 3e-4 \
        --seed $seed \
        --num-layers 4

    done

done

uv run python3 quick_plot.py --graph chain5 --context 200 --num-layers 4

uv run python3 quick_plot.py --graph tree5 --context 200 --num-layers 4

uv run python3 quick_plot.py --graph general --context 200 --num-layers 4

uv run python3 quick_plot.py --graph chain --context 200 --num-layers 4

uv run python3 quick_plot.py --graph tree --context 200 --num-layers 4

uv run python3 quick_plot.py --graph general7 --context 200 --num-layers 4
