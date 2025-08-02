import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import time
import os
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

'''
任务：
使用FashionMNIST数据集（已经预处理为ImageFolder形式）
训练一个softmax回归模型，并评估其性能。
'''

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
    print("加载FashionMNIST数据集...")
    data_dir = '../data/FashionMnist/images'

    # 定义数据转换
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 确保单通道
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # 单通道归一化
    ])

    # 加载完整训练集和测试集
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=transform
    )

    # 获取类别名称
    class_names = train_dataset.classes
    print(f"数据集包含 {len(class_names)} 个类别: {class_names}")

    # 划分训练集和验证集 (80%训练, 20%验证)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_set, val_set = random_split(train_dataset, [train_size, val_size])

    # ==================== 2. 模型配置 ====================

    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_set, batch_size=batch_size, pin_memory=(device.type == 'cuda'))
    # 对测试集打乱顺序，方便后面抽样看结果（一般来说不需要打乱顺序）
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda'))

    # 获取图像尺寸
    sample_image, _ = train_dataset[0]
    input_dim = sample_image.shape[1] * sample_image.shape[2]  # 高度 * 宽度
    output_dim = len(class_names)  # 类别数量

    print(f"图像尺寸: {sample_image.shape[1]}x{sample_image.shape[2]}")
    print(f"输入维度: {input_dim}, 输出维度: {output_dim}")

    learning_rate = 0.01
    num_epochs = 10

    # 定义模型,注意使用CrossEntropyLoss不需要额外的Softmax层
    model = nn.Sequential(
        nn.Flatten(),  # 将图像展平为向量
        nn.Linear(input_dim, output_dim)  # 直接输出logits
    ).to(device)

    # 定义损失函数：CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

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
            total_train += labels.size(0)  # 对批次的样本数量累加
            train_loss += loss.item()  # 对批次的平均loss累加
            _, predicted = torch.max(outputs.data, 1)  # 在维度1（类别）取最大值，返回的是元组（最大值tensor，索引tensor）
            correct_train += (predicted == labels).sum().item()  # 对批次的正确预测数量累加

        # 计算训练集指标（loss & acc）
        train_loss /= len(train_loader)  # 计算出每个样本的平均loss
        train_acc = correct_train / total_train  # 计算出训练集准确率
        train_losses.append(train_loss)  # 记录，方便绘图
        train_accs.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_val += labels.size(0)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == labels).sum().item()

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
            print(f"发现新的最佳模型! ")

        # 打印进度
        print(f"Epoch [{epoch + 1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # 保存最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), 'fashion_mnist_classifier.pth')
    print(f"训练完成! 总耗时: {time.time() - start_time:.1f}秒")
    print(f"模型已保存至: fashion_mnist_classifier.pth")

    # ==================== 4. 训练结果可视化 ====================
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    # ==================== 5. 测试集评估 ====================
    # 统计指标
    model.eval()
    correct_test = 0
    total_test = 0
    test_loss = 0.0

    # 记录预测结果
    test_predictions = []
    test_true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_test += labels.size(0)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_test += (predicted == labels).sum().item()

            test_predictions.extend(predicted.cpu().numpy())
            test_true_labels.extend(labels.cpu().numpy())

    test_acc = correct_test / total_test
    test_loss /= len(test_loader)

    print(f"\n测试集结果: Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'True Label': test_true_labels,
        'True Label Name': [class_names[i] for i in test_true_labels],
        'Predicted Label': test_predictions,
        'Predicted Label Name': [class_names[i] for i in test_predictions]
    })
    results_df['Correct'] = results_df['True Label'] == results_df['Predicted Label']
    results_df.to_csv('test_results.csv', index=False)
    print("测试结果已保存至 test_results.csv")

    # 计算并显示混淆矩阵
    cm = confusion_matrix(test_true_labels, test_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Reds', xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    # ==================== 6. 可视化部分测试样本 ====================
    # 获取一批测试样本
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    # 预测
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # 可视化预测结果
    plt.figure(figsize=(15, 10))
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        # 反归一化并转换为适合显示的格式
        image = images[i].cpu().squeeze()
        image = image * 0.5 + 0.5  # 反归一化
        image = image.numpy()

        # 如果是单通道，复制为三通道以便显示
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        plt.imshow(image)

        # 获取预测概率
        pred_prob = probabilities[i][predicted[i]].item() * 100

        # 设置标题颜色和内容
        true_label_idx = labels[i].item()
        pred_label_idx = predicted[i].item()
        color = 'green' if pred_label_idx == true_label_idx else 'red'
        title = (f"Pred: {class_names[pred_label_idx]} ({pred_prob:.1f}%)\n"
                 f"True: {class_names[true_label_idx]}")
        plt.title(title, color=color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('predictions_example.png')
    plt.show()

'''
结果评估：
在测试集上面平均准确率能够达到83%，比随机猜测的10%要好很多
而且观察抽样的预测结果，很多图片本身就很抽象，人类都很难分辨，所以模型能够达到这个准确率恐怕已经是极限
但这也侧面说明了，这个数据集本身很简单。
如果做更复杂的任务，就需要更复杂的模型
'''