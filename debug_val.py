import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

# ---------- 复用你的 MergedDataset ----------
class MergedDataset(Dataset):
    def __init__(self, images_tensor, labels_tensor, indices, transform=None):
        self.images = images_tensor
        self.labels = labels_tensor
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img = self.images[real_idx]
        lbl = self.labels[real_idx]
        if self.transform:
            img = self.transform(img)
        return img, lbl

# 加载数据
all_images = torch.load(r"D:\12blood\all_images.pt")
all_labels = torch.load(r"D:\12blood\all_labels.pt")
if isinstance(all_labels, list):
    all_labels = torch.tensor(all_labels)

total_len = len(all_labels)
indices = np.random.permutation(total_len)
train_end = int(0.6 * total_len)
val_end = int(0.7 * total_len)
val_idx = indices[train_end:val_end]

val_ds = MergedDataset(all_images, all_labels, val_idx, transform=None)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

# 1. 查看验证集标签分布
val_labels = all_labels[val_idx]
unique, counts = torch.unique(val_labels, return_counts=True)
print("验证集各类别样本数:")
for u, c in zip(unique, counts):
    print(f"  类别 {u}: {c} 张 ({c/len(val_labels)*100:.1f}%)")

# 2. 加载你刚训练出来的模型，看它在验证集上预测什么
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v3_small(weights=None, num_classes=13).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

all_preds = []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device).float()
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().tolist())

pred_tensor = torch.tensor(all_preds)
unique_p, counts_p = torch.unique(pred_tensor, return_counts=True)
print("\n模型在验证集上的预测类别分布:")
for u, c in zip(unique_p, counts_p):
    print(f"  预测类别 {u}: {c} 次 ({c/len(all_preds)*100:.1f}%)")