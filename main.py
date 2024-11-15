import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCDetection

from learn import learn
from MyModule import SSD

def main():
    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(p=0.5), # 左右反転
        transforms.RandomCrop(300, padding=8), # データの切り抜き
        transforms.RandomRotation(10), # 回転する角度の範囲を指定. ここで10とすると、-10度から+10度までの範囲でランダムに回転
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 0~1 => -1 ~ 1
    ])

    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 0~1 => -1 ~ 1
    ])



    def collate_fn(batch):
        images = []
        targets = []
        for item in batch:
            images.append(item[0])   # 画像データをリストに追加
            targets.append(item[1])  # アノテーション（辞書）をリストに追加
        return torch.stack(images, 0), targets  # 画像のみテンソル化し、アノテーションはリストのまま返す



    batch_size = 8
    train_dataset = VOCDetection(root='./dataset/voc_data', year='2012', image_set='train', \
                        download=True, transform=train_transform)
    val_dataset = VOCDetection(root='./dataset/val_voc_data', year='2012', image_set='val', \
                        download=True, transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)



    # モデルの定義
    ssd = SSD()
    opt = optim.Adam(ssd.parameters(), lr=0.03, weight_decay=1e-4)
    num_epochs = 10
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    num_classes = 21
    train_total_losses, val_total_losses, train_loc_losses, val_loc_losses, train_cls_losses, val_cls_losses = \
    learn(ssd, num_epochs, opt, train_loader, val_loader, num_classes=num_classes , save_path=None, early_stop=False, device=device)



if __name__ == "__main__":
    main()


