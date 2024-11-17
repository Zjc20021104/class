import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import torch.optim as optim
from tqdm import tqdm

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
#print("Classes:", train_dataset.classes)

# 输出每个图像的路径及其对应的类别标签，并查看Tensor的形状
#     for img_path, label in train_dataset.samples:
#         # 获取图像的Tensor形式
#         img_tensor = train_dataset[train_dataset.samples.index((img_path, label))][0]
#
#         print(img_tensor.shape)


# print(train_dataset[1][0])
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层卷积：输入通道数 3，输出通道数 16，卷积核大小 5x5
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层

        # 第二层卷积：输入通道数 16，输出通道数 32，卷积核大小 5x5
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        # 第三层卷积：输入通道数 32，输出通道数 64，卷积核大小 3x3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)

        # 全连接层：根据卷积输出的特征图尺寸计算输入大小
        self.fc1 = nn.Linear(64 * 6 * 6, 128)  # 假设分类任务
        self.fc2 = nn.Linear(128, 914)  # 输出分类数为 914

    def forward(self, x):
        # 第一层卷积+激活+池化
        x = self.pool(F.relu(self.conv1(x)))
        #print("After conv1:", x.shape)

        # 第二层卷积+激活+池化
        x = self.pool(F.relu(self.conv2(x)))
        #print("After conv2:", x.shape)

        # 第三层卷积+激活+池化
        x = self.pool(F.relu(self.conv3(x)))
        #print("After conv3:", x.shape)

        # 展平
        x = torch.flatten(x, 1)
        #print("After flatten:", x.shape)

        # 全连接层
        x = F.relu(self.fc1(x))
        #print("After fc1:", x.shape)
        x = self.fc2(x)
        #print("After fc2:", x.shape)

        return x


# 创建模型
net = Net()
#print(net)

# test = torch.randn((1, 3, 75, 75))
# pred = net(test)  # forward
class_indices = defaultdict(list)

# 获取每个类的所有样本索引
for idx, (_, label) in enumerate(train_dataset.samples):
    class_indices[label].append(idx)

train_samples = []
test_samples = []

# 为每个类进行划分
for label, indices in class_indices.items():
    # 计算每个类的 80% 和 20%
    num_samples = len(indices)
    num_train = int(0.8 * num_samples)
    num_test = num_samples - num_train

    # 打乱样本顺序
    np.random.shuffle(indices)

    # 选择训练集和测试集的索引
    train_samples.extend(indices[:num_train])
    test_samples.extend(indices[num_train:])

# 创建训练集和测试集
train_subset = torch.utils.data.Subset(train_dataset, train_samples)
test_subset = torch.utils.data.Subset(train_dataset, test_samples)

# 创建 DataLoader
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

# 查看数据集划分结果
print(f"Training samples: {len(train_samples)}")
print(f"Testing samples: {len(test_samples)}")

n_epochs = 10000  # How many passes through the training data
batch_size = 50  # Training batch size usually in [1,256]

learning_rate = 0.01  # Learning rate for optimizer like SGD usually in [0.001, 0.1]

random_seed = 1

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.5)

history = {'Train Loss': [], 'Test Loss': [], 'Test Accuracy': []}

for epoch in range(1, n_epochs + 1):
    # Create tqdm progress bar
    processBar = tqdm(train_loader, unit='step')
    # Set the network to training mode
    net.train(True)
    totalTrainLoss = 0.0

    for step, (trainImgs, labels) in enumerate(processBar):
        # Move images and labels to the device (GPU or CPU)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(trainImgs)

        # Calculate loss
        loss = criterion(outputs, labels)
        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(predictions == labels) / labels.shape[0]

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update progress bar
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" % (epoch, n_epochs, loss.item(), accuracy.item()))

        totalTrainLoss += loss

        if step == len(processBar) - 1:
            correct, totalLoss = 0, 0
            totalSize = 0
            net.train(False)
            for testImgs, labels in test_loader:
                outputs = net(testImgs)
                loss = criterion(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)
                totalSize += labels.size(0)
                totalLoss += loss
                correct += torch.sum(predictions == labels)
            testAccuracy = correct / totalSize
            testLoss = totalLoss / len(test_loader)
            trainLoss = totalTrainLoss / len(train_loader)
            history['Train Loss'].append(trainLoss.item())
            history['Test Loss'].append(testLoss.item())
            history['Test Accuracy'].append(testAccuracy.item())
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                       (epoch, n_epochs, loss.item(), accuracy.item(), testLoss.item(),
                                        testAccuracy.item()))
    processBar.close()
    torch.save(net.state_dict(), './models/class.pth')
