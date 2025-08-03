import numpy as np
import pandas as pd

'''
对数据进行预处理
1. 处理数值特征（先标准化，再填充缺失值）
2. 处理分类特征（one-hot编码）
'''

def normalize(x):
    # 处理标准差为0的情况
    std = x.std()
    if std == 0:  # 标准差为0时返回原值
        return x
    return (x - x.mean()) / std

# 加载数据
train_data = pd.read_csv('../data/house_price_prediction/train.csv')
test_data = pd.read_csv('../data/house_price_prediction/test.csv')

# 提取训练集标签和测试集ID
train_labels = train_data.SalePrice
test_ids = test_data.Id

# 先合并train,test一起处理（排除ID和标签列）
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 数值特征处理
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(normalize)
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# 分类特征处理(one-hot)
all_features = pd.get_dummies(all_features, dummy_na=True)

# 分割回训练集和测试集
n_train = len(train_data)
train_features = all_features[:n_train]
test_features = all_features[n_train:]

# 分别保存处理后的训练集和测试集
# 保存训练集（包含标签）
train_features_with_label = train_features.copy()
train_features_with_label['SalePrice'] = train_labels
train_features_with_label.to_csv('processed_train.csv', index=False)

# 保存测试集（包含ID）
test_features_with_id = test_features.copy()
test_features_with_id.insert(0, 'Id', test_ids.values)  # 添加ID列作为第一列
test_features_with_id.to_csv('processed_test.csv', index=False)
