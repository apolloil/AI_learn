import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import time

'''
任务：
使用线性回归模型结合梯度下降算法，
对房价数据(..\\data\\house_price_prediction）(注意：已经经过数据预处理)进行训练
由于dataset小(只有1千多)，使用k折交叉验证,从而调整超参数
'''

if __name__ == '__main__':
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 种子设置（结果确保可复现）
    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
    # ==================== 1. 数据加载 ====================
    train_df = pd.read_csv('..\\data\\house_price_prediction\\processed_train.csv')
    X = train_df.drop('SalePrice', axis=1).values
    y = train_df['SalePrice'].values.reshape(-1, 1)

    # 转换为张量并创建数据集
    full_dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

    # ==================== 2. 训练参数 ====================
    k = 10  # 交叉验证折数
    input_dim = X.shape[1]
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 400

    # ==================== 3. k折交叉验证训练 ====================
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = {'models': [], 'val_loss': []}

    print(f"开始进行{k}折交叉验证训练...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(full_dataset)):
        print(f"\n===== 第 {fold + 1} 折 =====")
        start_time = time.time()
        # 创建模型和优化器（SGD） - 每折使用新模型
        model = nn.Sequential(nn.Linear(input_dim, 1)).to(device)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # 创建数据加载器
        train_loader = DataLoader(
            full_dataset, batch_size=batch_size,
            sampler=SubsetRandomSampler(train_idx),
            pin_memory=(device.type == 'cuda')
            # 如果使用GPU，则使用pin_memory来锁定内存，加快数据传输
        )

        val_loader = DataLoader(
            full_dataset, batch_size=batch_size,
            sampler=SubsetRandomSampler(val_idx),
            pin_memory=(device.type == 'cuda')
        )

        # 训练循环
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            # 验证阶段
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)

            # 记录最佳验证损失
            avg_val_loss = val_loss / len(val_idx)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

            # 每20轮打印进度
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}")

        # 保存模型和结果
        fold_results['models'].append({k: v.cpu() for k, v in model.state_dict().items()})
        fold_results['val_loss'].append(best_val_loss)

        # 清理资源
        del model, optimizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        print(f"完成! 时间: {time.time() - start_time:.1f}秒, 最佳Val Loss: {best_val_loss:.4f}")

    # ==================== 4. 结果分析与模型选择 ====================
    # 选择最佳模型
    best_fold = np.argmin(fold_results['val_loss'])
    print(f"\n{k}折交叉验证总结:")
    print(f"选择第 {best_fold + 1} 折模型 (损失: {fold_results['val_loss'][best_fold]:.4f})")

    # 加载最佳模型
    best_model = nn.Sequential(nn.Linear(input_dim, 1)).to(device)
    best_model.load_state_dict({
        k: v.to(device) for k, v in fold_results['models'][best_fold].items()
    })

    # 保存模型
    torch.save(best_model.state_dict(), 'best_house_price_model.pth')
    print("最佳模型已保存为 best_house_price_model.pth")

    # ==================== 5. 可视化训练结果 ====================
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, k + 1), fold_results['val_loss'])
    plt.axhline(np.mean(fold_results['val_loss']), color='r', linestyle='--')
    plt.xlabel('Fold')
    plt.ylabel('Validation Loss')
    plt.title(f'{k}-Fold Cross Validation Results')
    plt.savefig('cv_results.png')
    plt.show()

    # ==================== 6. 测试集预测 ====================
    # 加载测试数据
    test_df = pd.read_csv('..\\data\\house_price_prediction\\processed_test.csv')
    test_ids = test_df['Id'].values
    X_test = test_df.drop('Id', axis=1).values

    # 进行预测
    best_model.eval()
    with torch.no_grad():
        predictions = best_model(
            torch.tensor(X_test, dtype=torch.float32).to(device)
        )
    predicted_prices = predictions.cpu().numpy().flatten()

    # 保存预测结果
    submission = pd.DataFrame({'Id': test_ids, 'SalePrice': predicted_prices})
    submission.to_csv('house_price_submission.csv', index=False)
    print("\n预测结果已保存至 house_price_submission.csv")

'''
结果评估：
在paddle上面rk 67%左右（RMSLE大约0.15左右）
可见模型效果一般，有进一步优化的空间
'''
