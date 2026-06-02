import torch
import os
from tqdm import tqdm

data_dir = r"D:\12blood\Dataset"          # 预处理输出文件夹（小 .pt 文件）
out_img = r"D:\12blood\all_images.pt"     # 合并后的图像
out_lbl = r"D:\12blood\all_labels.pt"     # 标签（直接从原始 labels.pt 复制）

# 读取已有标签
labels = torch.load(os.path.join(data_dir, "labels.pt"))
N = len(labels)
print(f"Total images: {N}")

# 逐张加载并堆叠
images_list = []
for idx in tqdm(range(N), desc="Merging images"):
    img = torch.load(os.path.join(data_dir, f"{idx}.pt"))  # float16 (3,128,128)
    images_list.append(img)

all_images = torch.stack(images_list)   # (N, 3, 128, 128) float16

# 保存
torch.save(all_images, out_img)
torch.save(labels, out_lbl)
print(f"Merged data saved to:\n  {out_img}\n  {out_lbl}")
print(f"Image tensor shape: {all_images.shape}, dtype: {all_images.dtype}")