import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import matplotlib.pyplot as plt
import time
import os
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import torch.nn.functional as F

'''
任务：
设计一个ResNet模型，
对CIFAR10数据集进行分类
'''

# 用于加载无标签测试集
class TestDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # 获取实际文件数量
        self.files = [f for f in os.listdir(root) if f.endswith('.png')]
        self.num_files = len(self.files)
        print(f"测试集包含 {self.num_files} 张图片")

    def __len__(self):
        return self.num_files

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, f'{idx + 1}.png')
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

# 残差块
class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 短路连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),   # 上1x1卷积(通道数目变化/图片尺寸变化)
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径前向传播
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 添加短路连接
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        # 初始设置
        self.current_channels = 64

        # 输入层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 残差层组
        self.layer1 = self._make_layer(64, 2, stride=1)  # 64×32×32
        self.layer2 = self._make_layer(128, 2, stride=2)  # 128×16×16
        self.layer3 = self._make_layer(256, 2, stride=2)  # 256×8×8
        self.layer4 = self._make_layer(512, 2, stride=2)  # 512×4×4

        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        # 创建包含多个残差块的层
        layers = []

        # 第一个块可能改变尺寸和通道
        layers.append(BasicBlock(self.current_channels, out_channels, stride))
        self.current_channels = out_channels  # 更新当前通道数

        # 后续块保持相同尺寸
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.current_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入层
        out = F.relu(self.bn1(self.conv1(x)))  # 64×32×32

        # 残差层组
        out = self.layer1(out)  # 64×32×32
        out = self.layer2(out)  # 128×16×16
        out = self.layer3(out)  # 256×8×8
        out = self.layer4(out)  # 512×4×4

        # 分类器
        out = self.avgpool(out)  # 512×1×1
        out = torch.flatten(out, 1)  # 512
        out = self.fc(out)  # 分类输出
        return out

if __name__ == '__main__':
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 种子设置（结果确保可复现）
    torch.manual_seed(42)
    np.random.seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)

    # ==================== 1. 数据预处理 ====================
    print("加载CIFAR10数据集...")

    # 定义数据转换 - 添加数据增强
    # 注意：transform不是预处理的，而是每次加载时应用，所以每个epoch数据会有不同
    train_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 验证和测试不需要数据增强
    test_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 加载训练集
    train_dataset = datasets.ImageFolder(
        root='../../data/cifar10/train',
        transform=train_transform  # 使用训练增强
    )

    # 获取类别名称
    class_names = train_dataset.classes
    print(f"数据集包含 {len(class_names)} 个类别: {class_names}")

    # 划分训练集和验证集 (80%训练, 20%验证)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_set, val_set = random_split(train_dataset, [train_size, val_size])

    # 加载测试集（无标签）
    test_dataset = TestDataset(
        root='../../data/cifar10/test',
        transform=test_transform
    )

    # ==================== 2. 模型配置 ====================

    # 创建数据加载器
    batch_size = 128
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             pin_memory=(device.type == 'cuda'))

    # 获取图像尺寸
    sample_image, _ = train_dataset[0]  # 右边返回的是一个元组，（图象数据，标签）
    input_dim = sample_image.shape[0] * sample_image.shape[1] * sample_image.shape[2]  # 通道*高度*宽度
    output_dim = len(class_names)  # 类别数量

    print(f"图像尺寸: {sample_image.shape[0]}x{sample_image.shape[1]}x{sample_image.shape[2]}")

    num_epochs = 30
    learning_rate = 0.01
    weight_decay = 5e-4


    # 定义模型
    model = ResNet().to(device)

    print("模型结构:")
    print(model)

    # 定义损失函数：CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=weight_decay)
    # 学习率调度器（原始衰减策略）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.98)

    # ==================== 3. 训练循环 ====================
    print("开始训练...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    best_model_state = None
    best_val_acc = 0.0

    start_time = time.time()
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计指标
            total_train += labels.size(0)
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == labels).sum().item()

        # 更新学习率（每个epoch结束时）
        scheduler.step()

        # 计算训练集指标
        train_loss /= len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        # 用于混淆矩阵的预测结果
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_val += labels.size(0)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()

                val_predictions.extend(predicted.cpu().numpy())
                val_true_labels.extend(labels.cpu().numpy())

        # 计算验证集指标
        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"Epoch {epoch + 1}: 发现新的最佳模型! Val Acc: {val_acc:.4f}")

        # 打印进度
        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ")

    # 保存最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), 'cifar10_classifier.pth')
    print(f"训练完成! 总耗时: {time.time() - start_time:.1f}秒")
    print(f"模型已保存至: cifar10_classifier.pth")

    # ==================== 4. 训练结果可视化 ====================
    plt.figure(figsize=(12, 10))

    # 损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 准确率曲线
    plt.subplot(2, 1, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    # 使用最佳模型在验证集上计算混淆矩阵
    model.eval()
    val_predictions = []
    val_true_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            val_predictions.extend(predicted.cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())

    # 计算并显示混淆矩阵
    cm = confusion_matrix(val_true_labels, val_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, cmap='Purples', xticks_rotation=45)
    plt.title(f'Validation Set Confusion Matrix (Acc: {best_val_acc:.4f})')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    # ==================== 5. 测试集预测 ====================
    # 进行测试集预测
    print(f"\n开始预测测试集 ({len(test_dataset)} 张图片)...")
    model.eval()
    predictions = []

    # 批量预测以提高效率
    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader):
            if isinstance(images, list):
                images = images[0]  # 如果DataLoader返回元组，取第一个元素

            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())

            # 打印进度
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(test_loader):
                print(f"已处理批次 [{batch_idx + 1}/{len(test_loader)}]")

    # 创建预测结果DataFrame
    test_results = pd.DataFrame({
        'id': [i + 1 for i in range(len(test_dataset))],
        'label': [class_names[l] for l in predictions]
    })

    # 保存预测结果
    test_results.to_csv('test_predictions.csv', index=False)
    print(f"\n测试集预测结果已保存至: test_predictions.csv")

    print("所有任务完成!")

'''
结果评估：
ResNet在训练的过程中，仅仅6个epoch就能拿到79%的准确率，与训练30个epoch的GoogLenet相当
在30个epoch后，准确率达到了86.41%，这样简单的一个模型，效果比之前任何一个模型都要好
ResNet真是天才之作
'''