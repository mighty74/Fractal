DATA=fractal1k
DATAROOT=/home/white/make_cross_valid_data/dataset_kindai/normal/num1
CKPT=/home/white/FractalDB-Pretrained-ViT-PyTorch/outputs/2023-07-07/22-51-19/deit_tiny_patch16_224_fractal1k_090ep.pth

# num1
python finetune.py \
ckpt=$CKPT \
data=$DATA \
data.set.root=$DATAROOT \
data.transform.re_prob=0 \
data.loader.batch_size=96 \
model=deit_tiny_patch16_224 \
model.drop_path_rate=0.0 \
optim=momentum \
optim.args.lr=0.01 \
optim.args.weight_decay=1.0e-4 \
scheduler.args.warmup_epochs=10


# num2
DATAROOT=/home/white/make_cross_valid_data/dataset_kindai/normal/num2

python finetune.py \
ckpt=$CKPT \
data=$DATA \
data.set.root=$DATAROOT \
data.transform.re_prob=0 \
data.loader.batch_size=96 \
model=deit_tiny_patch16_224 \
model.drop_path_rate=0.0 \
optim=momentum \
optim.args.lr=0.01 \
optim.args.weight_decay=1.0e-4 \
scheduler.args.warmup_epochs=10



# num3
DATAROOT=/home/white/make_cross_valid_data/dataset_kindai/normal/num3

python finetune.py \
ckpt=$CKPT \
data=$DATA \
data.set.root=$DATAROOT \
data.transform.re_prob=0 \
data.loader.batch_size=96 \
model=deit_tiny_patch16_224 \
model.drop_path_rate=0.0 \
optim=momentum \
optim.args.lr=0.01 \
optim.args.weight_decay=1.0e-4 \
scheduler.args.warmup_epochs=10



# num4
DATAROOT=/home/white/make_cross_valid_data/dataset_kindai/normal/num4

python finetune.py \
ckpt=$CKPT \
data=$DATA \
data.set.root=$DATAROOT \
data.transform.re_prob=0 \
data.loader.batch_size=96 \
model=deit_tiny_patch16_224 \
model.drop_path_rate=0.0 \
optim=momentum \
optim.args.lr=0.01 \
optim.args.weight_decay=1.0e-4 \
scheduler.args.warmup_epochs=10



# num5
DATAROOT=/home/white/make_cross_valid_data/dataset_kindai/normal/num5

python finetune.py \
ckpt=$CKPT \
data=$DATA \
data.set.root=$DATAROOT \
data.transform.re_prob=0 \
data.loader.batch_size=96 \
model=deit_tiny_patch16_224 \
model.drop_path_rate=0.0 \
optim=momentum \
optim.args.lr=0.01 \
optim.args.weight_decay=1.0e-4 \
scheduler.args.warmup_epochs=10



# num6
DATAROOT=/home/white/make_cross_valid_data/dataset_kindai/normal/num6

python finetune.py \
ckpt=$CKPT \
data=$DATA \
data.set.root=$DATAROOT \
data.transform.re_prob=0 \
data.loader.batch_size=96 \
model=deit_tiny_patch16_224 \
model.drop_path_rate=0.0 \
optim=momentum \
optim.args.lr=0.01 \
optim.args.weight_decay=1.0e-4 \
scheduler.args.warmup_epochs=10



# num7
DATAROOT=/home/white/make_cross_valid_data/dataset_kindai/normal/num7

python finetune.py \
ckpt=$CKPT \
data=$DATA \
data.set.root=$DATAROOT \
data.transform.re_prob=0 \
data.loader.batch_size=96 \
model=deit_tiny_patch16_224 \
model.drop_path_rate=0.0 \
optim=momentum \
optim.args.lr=0.01 \
optim.args.weight_decay=1.0e-4 \
scheduler.args.warmup_epochs=10



# num8
DATAROOT=/home/white/make_cross_valid_data/dataset_kindai/normal/num8

python finetune.py \
ckpt=$CKPT \
data=$DATA \
data.set.root=$DATAROOT \
data.transform.re_prob=0 \
data.loader.batch_size=96 \
model=deit_tiny_patch16_224 \
model.drop_path_rate=0.0 \
optim=momentum \
optim.args.lr=0.01 \
optim.args.weight_decay=1.0e-4 \
scheduler.args.warmup_epochs=10



# num9
DATAROOT=/home/white/make_cross_valid_data/dataset_kindai/normal/num9

python finetune.py \
ckpt=$CKPT \
data=$DATA \
data.set.root=$DATAROOT \
data.transform.re_prob=0 \
data.loader.batch_size=96 \
model=deit_tiny_patch16_224 \
model.drop_path_rate=0.0 \
optim=momentum \
optim.args.lr=0.01 \
optim.args.weight_decay=1.0e-4 \
scheduler.args.warmup_epochs=10



# num10
DATAROOT=/home/white/make_cross_valid_data/dataset_kindai/normal/num10

python finetune.py \
ckpt=$CKPT \
data=$DATA \
data.set.root=$DATAROOT \
data.transform.re_prob=0 \
data.loader.batch_size=96 \
model=deit_tiny_patch16_224 \
model.drop_path_rate=0.0 \
optim=momentum \
optim.args.lr=0.01 \
optim.args.weight_decay=1.0e-4 \
scheduler.args.warmup_epochs=10
