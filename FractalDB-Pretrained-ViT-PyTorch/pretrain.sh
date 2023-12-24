DATAROOT=./data/FractalDB-1000/00000

python pretrain.py \
data.set.root=$DATAROOT \
model=deit_tiny_patch16_224 \
optim.args.lr=3.0e-4 \
scheduler.args.warmup_epochs=10
