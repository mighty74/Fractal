# DATA=fractal1k
DATA=cifar10
# train data
# DATAROOT=/home/white/daikon_kindai/num1/train
DATAROOT=/home/white/daikon_kindai/num1/train
CKPT=/home/white/FractalDB-Pretrained-ViT-PyTorch/outputs/2023-12-04/10-37-42/deit_tiny_patch16_224_fractal1k_100ep.pth

# test dta
# path2db=/home/white/daikon_kindai/num1
path2db=/home/white/daikon_kindai/num1

python3 finetune_eval.py \
ckpt=$CKPT \
data=$DATA \
data.set.root=$DATAROOT \
data.transform.re_prob=0 \
data.loader.batch_size=32 \
model=deit_tiny_patch16_224 \
model.drop_path_rate=0.0 \
optim=momentum \
optim.args.lr=0.1 \
optim.args.weight_decay=1.0e-4 \
scheduler.args.warmup_epochs=10 \
path2db=$path2db
# optim.args.lr=0.01 \
