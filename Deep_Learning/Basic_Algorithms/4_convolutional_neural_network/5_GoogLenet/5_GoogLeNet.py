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

'''
任务：
设计一个GoogLeNet模型，
对CIFAR10数据集进行分类
注：v1版本的googlenet没有BN，这里为了减少调参工作，引入BN层进行优化
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

# Inception模块
class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super().__init__()

        # 分支1: 1x1卷积
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        # 分支2: 1x1卷积 -> 3x3卷积
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )

        # 分支3: 1x1卷积 -> 5x5卷积
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5, ch5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )

        # 分支4: 3x3最大池化 -> 1x1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # 在通道维度拼接所有分支的输出
        return torch.cat([branch1, branch2, branch3, branch4], 1)

# 辅助分类器，注入梯度到中间层加快训练
class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, feat_size=4):
        super().__init__()
        # 使用自适应池化处理不同尺寸
        self.avg_pool = nn.AdaptiveAvgPool2d((feat_size, feat_size))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        # 动态计算全连接层输入
        fc_input = 128 * feat_size * feat_size
        self.fc1 = nn.Linear(fc_input, 512)
        self.dropout = nn.Dropout(0.7)
        self.fc2 = nn.Linear(512, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10, aux_logits=True):
        super().__init__()
        self.aux_logits = aux_logits

        # 修改初始卷积层（适配32x32）
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # 改为3x3卷积
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 移除第二个卷积层中的池化
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        # 减少Inception模块数量（适配小数据集）
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 下采样

        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 下采样

        # 修改辅助分类器输入尺寸
        if self.aux_logits:
            self.aux1 = AuxiliaryClassifier(512, num_classes, feat_size=8)  # 适配8x8特征图
            self.aux2 = AuxiliaryClassifier(512, num_classes, feat_size=8)

        # 修改主分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # GAP 全局平均池化
        self.dropout = nn.Dropout(0.4)
        # 输入通道数减少为512
        self.fc = nn.Linear(512, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入: 3×32×32
        x = self.conv1(x)  # → 64×32×32
        x = self.conv2(x)  # → 192×32×32

        # Inception 模块1
        x = self.inception3a(x)  # → 256×32×32
        x = self.inception3b(x)  # → 480×32×32
        x = self.maxpool1(x)  # → 480×16×16

        # Inception 模块2
        x = self.inception4a(x)  # → 512×16×16

        # 辅助分类器1
        aux1 = None
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)  # → 512×16×16
        x = self.maxpool2(x)  # → 512×8×8

        # 辅助分类器2
        aux2 = None
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        # 主分类器
        x = self.avgpool(x)  # → 512×1×1
        x = torch.flatten(x, 1)  # → 512
        x = self.dropout(x)
        x = self.fc(x)  # → num_classes

        if self.aux_logits and self.training:
            return x, aux1, aux2
        return x

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
    model = GoogLeNet().to(device)

    print("模型结构:")
    print(model)

    # 定义损失函数：CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=weight_decay)
    # 学习率调度器（原始衰减策略）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.96)

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
            # 如果是训练阶段，处理辅助分类器
            if model.training and model.aux_logits:
                main_output, aux1_output, aux2_output = outputs
                loss1 = criterion(main_output, labels)
                loss2 = criterion(aux1_output, labels)
                loss3 = criterion(aux2_output, labels)
                loss = loss1 + 0.3 * loss2 + 0.3 * loss3  # 加权和
            else:
                loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计指标
            total_train += labels.size(0)
            train_loss += loss.item()
            _, predicted = torch.max(outputs[0].data, 1)
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
79.5%左右，跟VGG相当，但略逊于NiN网络。
主要原因在于GoogleNet的Inception模块更适合于细粒度分类任务（不同的branch感受野不同）
而cifar10分辨率较小，更适用于高效特征抽象的NiN网络。
若在ImageNet上面做实验，Googlenet应该会表现出更好的效果。
'''