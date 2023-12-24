# DATAROOT=/home/white/daikon_kindai/sum
DATAROOT=/home/white/FractalDB-Pretrained-ViT-PyTorch/configs/data/FractalDB-1000

python3 pretrain.py \
data.set.root=$DATAROOT \
model=deit_tiny_patch16_224 \
optim.args.lr=3e-5 \
scheduler.args.warmup_epochs=10
