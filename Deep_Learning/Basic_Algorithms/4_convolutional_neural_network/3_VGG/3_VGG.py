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
使用VGG16Net
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

# 简化版本的VGG16
class VGG16Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # 卷积块1：2层卷积 + 最大池化
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        )

        # 卷积块2：2层卷积 + 最大池化
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        )

        # 卷积块3：3层卷积 + 最大池化
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4
        )

        # 卷积块4：3层卷积 + 最大池化
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 4x4 -> 2x2
        )

        # 分类器部分（适当缩小全连接层）
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),  # 2x2特征图
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用高斯分布初始化
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BN层使用常数初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层使用高斯分布初始化
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
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
    learning_rate = 0.1
    weight_decay = 5e-4


    # 定义模型
    model = VGG16Net().to(device)

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
30轮训练后测试集准确率79.48%左右，相比Alex又提升了7个百分点
可以看出VGG的意义：提供了一种设计CNN网络架构的范式，同时不断加深了模型，达到较高的精度
但是缺点也很明显，计算量明显比Alex大很多，训练时长增加不少（至少多出三倍的训练时间）
分析计算量的来源：
卷积层: k*k*C_in*C_out*H_out*W_out
全连接层: C_in*C_out
以VGG16为例，虽然卷积层参数只占10%左右，但是计算量却很大(2.16GFlOPS，占总计算量95%)

此外，去掉全部BN层（原始VGG模型）后，出现了跟Lenet类似的情况，始终准确率在10%左右，可见调参在BN、adam之前的困难性
'''