import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义预处理流程，将图像转换为Tensor
transform = transforms.Compose([

    transforms.ToTensor(),  # 将图像转换为Tensor
])

# 第一步：加载数据集，并将图像转换为Tensor
train_dir = './jin_wen/'
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)

# 打印类别标签
print("Classes:", train_dataset.classes)

# 输出每个图像的路径及其对应的类别标签，并查看Tensor的形状
for img_path, label in train_dataset.samples:
    # 获取图像的Tensor形式
    img_tensor = train_dataset[train_dataset.samples.index((img_path, label))][0]

    # 打印图像路径、标签和Tensor的形状
    print(f"Image Path: {img_path}, Label: {train_dataset.classes[label]}, Tensor Shape: {img_tensor.shape}")
