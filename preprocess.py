import torch, os
from torchvision import datasets, transforms
from tqdm import tqdm

src_dir = r"D:\12blood\cropped_cells"
out_dir = r"D:\12blood\Dataset"      # 128x128 足够用
os.makedirs(out_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((128, 128)),       # 🔥 降低分辨率，立竿见影
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

ds = datasets.ImageFolder(src_dir)
for idx, (img, lbl) in enumerate(tqdm(ds)):
    tensor = transform(img).half()
    torch.save(tensor, os.path.join(out_dir, f"{idx}.pt"))

torch.save(ds.targets, os.path.join(out_dir, "labels.pt"))
torch.save(ds.class_to_idx, os.path.join(out_dir, "class_to_idx.pt"))