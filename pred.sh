source .venv/bin/activate

python train_pred.py --graph tree --context-len 500 --num-layers 4 --train-size 1000

python train_pred.py --graph chain --context-len 500 --num-layers 4 --train-size 1000

python train_pred.py --graph general --context-len 500 --num-layers 4 --train-size 1000
