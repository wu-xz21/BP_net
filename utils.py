import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error


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