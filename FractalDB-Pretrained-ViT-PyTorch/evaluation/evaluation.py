import hydra
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from loadDB import DBLoader
from hydra.utils import instantiate
import models
from utils import fix_random_seed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



@hydra.main(config_path='../configs', config_name='evaluation')
def main(cfg):
    if cfg.seed is not None:
        fix_random_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    use_cuda = not cfg.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    # ファインチューニングしたモデルをロード
    print("model load")
    
    model = instantiate(cfg.model, num_classes=2)
    ckpt = torch.load(cfg.ckpt, map_location=device)
    ckpt_model = ckpt['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias', 'fc.weight', 'fc.bias']:
        if k in ckpt_model and ckpt_model[k].shape != state_dict[k].shape:
            print(f'Remove key [{k}] from pretrained checkpoint')
            del ckpt_model[k]
    model.load_state_dict(ckpt_model)

    model = torch.nn.parallel.DataParallel(model)
    model.cuda()
    model_without_dp = model.module

    # model_without_dp.load_state_dict(ckpt_model)
    # # load optimizer
    optimizer = instantiate(cfg.optim, model=model)
    optimizer.load_state_dict(ckpt['optimizer'])

    # # load lr_schedulerfg
    lr_scheduler, _ = instantiate(cfg.scheduler, optimizer=optimizer)
    lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

    loss_scaler = ckpt['scaler']
    epoch = ckpt['epoch']

    model.to(device)
    print(f'Checkpoint was loaded from {cfg.ckpt}\n')
    # model = torch.nn.parallel.DataParallel(model)
    model.eval()

    # model = instantiate(cfg.model, num_classes=2)
    # ckpt = torch.load(cfg.ckpt, map_location='cpu')
    # model.load_state_dict(ckpt, strict=False)
    # print(f'Checkpoint was loaded from {cfg.ckpt}\n')
    # model.cuda()
    # model.eval()




    # テストデータの準備
    print("load test data")
    test_transform = transforms.Compose([
						transforms.Resize(256, interpolation=2),
						transforms.CenterCrop(224),
						transforms.ToTensor(),
						transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    test_dataset = DBLoader(cfg.path2db,'test', test_transform)  # テストデータセットを適切に用意する
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # テストデータに対する予測と真のラベルの記録
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for i_batch, sample_batched in enumerate(test_loader):
            images, labels = sample_batched["image"].to(device), sample_batched["label"].to(device)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 評価指標の計算
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')

    print(f"Accuracy: {accuracy:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1 Score: {f1:.5f}")

    print(all_predictions)



if __name__ == '__main__':
    main()
