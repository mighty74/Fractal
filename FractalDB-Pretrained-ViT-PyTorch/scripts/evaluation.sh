# path2db={/path/to/testdb}
# CKPT=outputs/{pretrain_date}/{filename}.pth

path2db=/home/white/cifer/data/cifar-10
CKPT=~/FractalDB-Pretrained-ViT-PyTorch/outputs/2023-11-10/16-35-37/deit_tiny_patch16_224_fractal1k_200ep.pth

python evaluation/evaluation.py \
ckpt=$CKPT \
path2db=$path2db \
model=deit_tiny_patch16_224 \
model.drop_path_rate=0.0 \
optim=momentum \
optim.args.lr=0.01 \
optim.args.weight_decay=1.0e-4 \
scheduler.args.warmup_epochs=10
