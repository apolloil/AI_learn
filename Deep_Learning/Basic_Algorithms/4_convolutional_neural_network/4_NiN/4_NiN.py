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
使用NiN网络
对CIFAR10数据集进行分类
注：原论文初始化方法为高斯初始化，这里使用Kaiming初始化
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

# NiN网络
class NiN(nn.Module):
    def __init__(self, num_classes=10, drop_rate=0.5):
        super().__init__()

        # Block 1: input 3×32×32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),  # → 192×32×32
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1),  # → 160×32×32
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=1),  # → 96×32×32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # → 96×16×16
        )

        # Block 2: input 96×16×16
        self.block2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),  # → 192×16×16
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),  # → 192×16×16
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),  # → 192×16×16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # → 192×8×8
        )

        # Block 3: input 192×8×8
        self.block3 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),  # → 192×8×8
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),  # → 192×8×8
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),  # → 192×8×8
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),  # → 192×4×4
        )

        # Classifier: input 192×4×4
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),  # → 192×4×4
            nn.Conv2d(192, num_classes, 1),  # → 10×4×4
            nn.ReLU(inplace=True),  # → 10×4×4
            nn.AdaptiveAvgPool2d(1)  # → 10×1×1
        )

        # 初始化权重 - 使用稳定的Kaiming初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用Kaiming初始化替代原始高斯初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.normal_(m.weight, std=0.01)  # 原始高斯初始化
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入: 3×32×32
        x = self.block1(x)  # → 96×16×16
        x = self.block2(x)  # → 192×8×8
        x = self.block3(x)  # → 192×4×4
        x = self.classifier(x)  # → 10×1×1
        return x.view(x.size(0), -1)  # 展平为10维向量

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
    model = NiN().to(device)

    print("模型结构:")
    print(model)

    # 定义损失函数：CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # 添加权重衰减
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=weight_decay)

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
81.92%的准确率,比VGG进一步提高
然而高斯初始化还是一直只能10%？根本无法收敛?调参难度极高！
'''