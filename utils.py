from math import isinf

import joblib
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sko.PSO import PSO

# 1. BP神经网络类
class BP_NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2,hidden_size3, output_size):
        super(BP_NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)  # 输入层到第一隐藏层
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # 第一隐藏层到第二隐藏层
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)  # 第二隐藏层到输出层
        self.fc4 = nn.Linear(hidden_size3, output_size)  # 第二隐藏层到输出层
        self.relu = nn.ReLU()  # 激活函数
        self.sigmoid = nn.Sigmoid() #

    def forward(self, x):
        x = self.fc1(x)  # 输入通过第一层
        x = self.relu(x)  # 激活函数
        x = self.fc2(x)  # 通过第二层
        x = self.relu(x)  # 激活函数
        x = self.fc3(x)  # 通过第三层
        x = self.relu(x)  # 激活函数
        x = self.fc4(x)  # 通过第三层
        x = self.sigmoid(x)  # 最后一层，保证输出在0~1
        return x

def generate_samples(num_samples=1000, num_features=8):
    X = np.random.uniform(-1, 1, (num_samples, num_features))  # 8个输入参数，范围[-1, 1]
    y = np.sum(X, axis=1)  # 输出是输入参数的简单线性组合，例如y = x1 + x2 + ... + x8
    return X, y

# 2. 数据集划分
def split_data(X, y, test_size=0.2, val_size=0.2):
    # 划分训练集和测试集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    # 划分验证集和测试集
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42)

    return X_train, X_val, X_test, y_train, y_val, y_test

# 训练与评估
def train_and_validate(model, train_loader, val_loader, optimizer, criterion, num_epochs=100, patience=10):
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()  # 切换到训练模式
        running_loss = 0.0

        # 训练
        for inputs, targets in train_loader:
            optimizer.zero_grad()  # 清除之前的梯度
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            running_loss += loss.item()

        # 验证
        model.eval()  # 切换到评估模式
        val_loss = 0.0
        with torch.no_grad():  # 不计算梯度
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}')

        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

# 5. 模型评估
def evaluate_model(model, test_loader):
    model.eval()  # 切换到评估模式
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            y_true.extend(targets.numpy())
            y_pred.extend(outputs.numpy())

    # 计算均方误差
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    error = np.mean(np.abs((np.array(y_true)-np.array(y_pred))))
    print(f'Test MSE: {mse:.4f}')
    print(f'Test mean error:{error:.4f}')
    print(f'Test RMSE:{rmse:.4f}')

# 6. mat导入
def data_load(mat):
    mat_data = scipy.io.loadmat(mat)
    data = mat_data['data']
    x = data[:,:8]
    y = data[:,8]
    # 假设 X是输入，Y是输出
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    # 按照打乱后的顺序重排
    x_shuffled = x[indices]
    y_shuffled = y[indices]

    return x_shuffled,y_shuffled


# 约束条件
def monte_carlo_pass(x,optimize_T7,optimize_T8,model,scaler_x,target = 0.8):
    # 如果不能通过，就返回1
    # target: ACDC下限值, 比如98%的ACDC大于0.8
    lower_bounds = np.array([0,0,0,0,-x[4],-x[5],-optimize_T7,-optimize_T8])
    upper_bounds = np.array([x[0],x[1],x[2],x[3],0,0,optimize_T7,optimize_T8])
    mean = (lower_bounds+upper_bounds)/2
    std_dev = (upper_bounds - lower_bounds) / 4
    # 蒙特采样次数
    n_samples = 30000
    # 生成样本：形状 (n_samples, 8)
    np.random.seed(2024)    # 固定采样种子
    samples = np.random.normal(loc=mean, scale=std_dev, size=(n_samples, 8))
    # ------------------- 标准化输入 ------------------- #
    samples_scaled = scaler_x.transform(samples)
    # 转为PyTorch张量
    samples_tensor = torch.tensor(samples_scaled, dtype=torch.float32)
    # ------------------- 批量预测 ------------------- #
    with torch.no_grad():
        predictions = model(samples_tensor)
    count = torch.sum(predictions > target).item()
    pass_rate = count/n_samples
    # print("pass_rate:",pass_rate)
    # 统计蒙特卡洛通过率，如果大于98%则计该公差带有效
    if pass_rate>0.95:
        # print("pass")
        return 0
    else:
        return 1


# 代价函数
def loss(x):
    C_t = np.exp(-x[0])+np.exp(-x[1])+np.exp(-x[2])+np.exp(-x[3])+np.exp(-x[4])+np.exp(-x[5])
    # print('Loss:',C_t)
    mean = np.mean(x)
    alpha = 5
    # 方差作为均衡性惩罚项，越均匀越小
    balance_penalty = np.var(x)
    return C_t + alpha * balance_penalty

def tolerance(model, scaler_x, optimize_T7, optimize_T8):

    # 定义参数的维度和上下界
    dim = 6  # T1~T6
    unit_trans_scaler = 0.001*180/np.pi
    lb = np.array([0.05*unit_trans_scaler]*dim)  # 下界：公差不能小于0.05，换成角度
    ub = unit_trans_scaler*np.array([0.5,0.5,0.6,0.25,0.6,0.2])   # 上界：取自于初始公差带
    constraint_ueq = (lambda x: monte_carlo_pass(x, unit_trans_scaler*optimize_T7, unit_trans_scaler*optimize_T8, model, scaler_x, target=0.8),)

    pso = PSO(func=loss, n_dim=dim, pop=100, max_iter=200,
              lb=lb, ub=ub, w=0.8, c1=2, c2=2, constraint_ueq=constraint_ueq)
    pso.run()
    if isinf(pso.best_y):
        raise OverflowError("请调整参数上下界ub和lb的范围，避免出现无穷大的情况")

    # plt.plot(pso.gbest_y_hist)
    # plt.title('PSO Result')
    # plt.xlabel('Iteration', fontsize=14)
    # plt.ylabel('Loss(s)', fontsize=14)
    # plt.grid()
    # plt.show()
    # 转成弧度
    X = pso.gbest_x/unit_trans_scaler
    y = pso.gbest_y
    print('best_x is ', X, 'best_y is', y)
    return X, y
