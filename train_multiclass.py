import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns   # 让混淆矩阵更好看，如果没有可 pip install seaborn

# ==================== 全局设置 ====================
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==================== 自定义 Dataset ====================
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

# ==================== 加载数据 ====================
print("Loading merged data into RAM...")
all_images = torch.load(r"D:\12blood\all_images.pt")
all_labels = torch.load(r"D:\12blood\all_labels.pt")
if isinstance(all_labels, list):
    all_labels = torch.tensor(all_labels)

num_classes = len(torch.unique(all_labels))
total_len = len(all_labels)
print(f"Total samples: {total_len}, classes: {num_classes}")

# ---------- 划分 ----------
indices = np.random.permutation(total_len)
train_end = int(0.6 * total_len)
val_end   = int(0.7 * total_len)
train_idx = indices[:train_end]
val_idx   = indices[train_end:val_end]
test_idx  = indices[val_end:]

# ---------- 在线增强 ----------
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])

train_ds = MergedDataset(all_images, all_labels, train_idx, transform=train_transform)
val_ds   = MergedDataset(all_images, all_labels, val_idx,   transform=None)
test_ds  = MergedDataset(all_images, all_labels, test_idx,  transform=None)

batch_size = 128
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, pin_memory=True)

# ==================== 模型（预训练权重 + 替换分类头）====================
model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
model = model.to(device)
print(f"Model: MobileNetV3-Small (pretrained), params: {sum(p.numel() for p in model.parameters()):,}")

# ---------- 类别平衡的损失函数 ----------
train_labels = all_labels[train_idx]
class_counts = torch.bincount(train_labels, minlength=num_classes)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum() * num_classes
print(f"Class weights (balanced): {class_weights.tolist()}")
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
scaler = GradScaler('cuda')

# ==================== 训练（6 epoch）====================
epochs = 4
train_losses, val_accs = [], []
best_val_acc = 0.0

for epoch in range(epochs):
    # --- 训练 ---
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast('cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        loop.set_postfix(loss=loss.item())

    train_loss = total_loss / len(train_ds)
    train_acc = correct / total
    train_losses.append(train_loss)

    # --- 验证 ---
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device).float(), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()
    val_acc = val_correct / val_total
    val_accs.append(val_acc)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
    scheduler.step()

# ==================== 测试 + 混淆矩阵 ====================
print("\nLoading best model for testing...")
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

all_preds = []
all_targets = []
test_correct, test_total = 0, 0

with torch.no_grad():
    for imgs, labels in tqdm(test_loader, desc="Testing"):
        imgs, labels = imgs.to(device).float(), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (preds == labels).sum().item()

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

test_acc = test_correct / test_total
print(f"\n===== Final Test Accuracy: {test_acc:.4f} =====")

# ---------- 混淆矩阵 ----------
cm = confusion_matrix(all_targets, all_preds)
print("\nClassification Report:")
print(classification_report(all_targets, all_preds, digits=3))

# 可视化混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(num_classes),
            yticklabels=range(num_classes))
plt.title(f'Confusion Matrix (Test Acc: {test_acc:.3f})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# ==================== 绘制训练曲线 ====================
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, 'b-o', label='Train Loss')
plt.title('Training Loss')
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), val_accs, 'r-o', label='Val Accuracy')
plt.title('Validation Accuracy')
plt.tight_layout()
plt.savefig('training_curves.png')
plt.show()

print(f"Training completed. Best val accuracy: {best_val_acc:.4f}, Test accuracy: {test_acc:.4f}")