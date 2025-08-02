

# 简单的回归

## 线性回归  

### 网络架构

Elementwise form：  
对于单个输入，我们有
$$
\hat{y} = \mathbf{x}^\top \mathbf{w} + b
$$

$$
其中\mathbf{x}表示特征向量，\mathbf{w}表示权重向量
$$

Matrix-vector form：  
由于for循环开销巨大，进行矩阵向量形式的改写，不仅形式简洁，而且硬件可加速计算  
对于多个输入，我们有
$$
\hat{\mathbf{y}}=\mathbf{X}\mathbf{w}+b
$$

$$
其中\mathbf{X}的第i行表示第i个样本的特征向量
$$

### 损失函数

我们使用均方差损失函数(MSEloss)来定义损失函数，它是关于w,b的函数  
$$
\mathbf{L}(w,b)=\frac{(\hat{\mathbf{y}}-\mathbf{y})^2}{2\mathcal{N}}
$$

$$
其中\mathcal{N}是数据量大小,也就是\mathbf{X}的行数
$$

可以证明，若残差噪声服从高斯分布，当L最小时，整个数据集达到最大的似然

### 求解方法

最优化模型等价于最小化损失函数  
解析解：  
利用线性代数的知识，我们可以获得解析解如下：

$$
令\mathbf{A}=\begin{bmatrix} \mathbf{X} & \mathbf{1} \end{bmatrix}, \theta=\begin{bmatrix} \mathbf{w} \\ b \end{bmatrix}
$$

$$
那么问题等价于求\mathbf{y}=\mathbf{A}\theta的最小二乘解
$$

$$
从而可得\theta=(\mathbf A^\top \mathbf A)^{-1}\mathbf A^\top \mathbf{y}
$$

优化解：  
解析解虽然准确，但是在数据量很大的时候计算复杂，仅适用于小数据  
我们常用梯度下降（Gradient Descent）的方法找到优化解  
$$
每次取一部分大小为\mathcal{B}的数据,对参数\mathbf{w},b进行更新
$$

$$
(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \eta\times\frac{\partial{L}}{\partial(\mathbf{w},b)}
$$

$$
其中\eta是学习率
$$





