import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def pad_to_target_size(img, target_size):
    width, height = img.size  # 获取原始图像尺寸

    # 计算需要填充的像素数
    pad_left = (target_size - width) // 2
    pad_top = (target_size - height) // 2
    pad_right = target_size - width - pad_left
    pad_bottom = target_size - height - pad_top

    # 使用 padding 填充图像，填充色为0 (黑色)
    return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

# 定义预处理流程，将图像转换为Tensor
transform = transforms.Compose([
    transforms.Lambda(lambda img: pad_to_target_size(img, 75)),
    transforms.ToTensor(),  # 将图像转换为Tensor
])

# 第一步：加载数据集，并将图像转换为Tensor
train_dir = './jin_wen/'
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)

# 打印类别标签
# print("Classes:", train_dataset.classes)
print(len(train_dataset.classes))
class_num = len(train_dataset.classes)

# 输出每个图像的路径及其对应的类别标签，并查看Tensor的形状
# for img_path, label in train_dataset.samples:
#     # 获取图像的Tensor形式
#     img_tensor = train_dataset[train_dataset.samples.index((img_path, label))][0]
#     print(img_tensor.shape)

