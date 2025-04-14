import torch
import torch.nn as nn
import torch.optim as optim

from utils import generate_samples, split_data, BP_NeuralNetwork, train_and_validate, evaluate_model, data_load
import joblib
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# 生成样本数据
# X, y = generate_samples(num_samples=1000, num_features=8)
mat = './mat/hyper_cube_data.mat'
X, y = data_load(mat)

scaler_x = StandardScaler()
X_scaled = scaler_x.fit_transform(X)

# 划分数据集
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_scaled, y)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 创建数据加载器
train_data = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
val_data = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
test_data = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=16)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=16)

# 定义模型、损失函数和优化器
input_size = 8  # 输入特征数量
hidden_size1 = 10  # 第一隐藏层神经元数量
hidden_size2 = 10  # 第二隐藏层神经元数量
hidden_size3 = 10
output_size = 1  # 输出是一个数值
model = BP_NeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练和验证模型
train_and_validate(model, train_loader, val_loader, optimizer, criterion, num_epochs=100, patience=10)

# 评估模型在测试集上的表现
evaluate_model(model, test_loader)

# 保存模型参数
# 保存模型权重
torch.save(model.state_dict(), 'bp_neural_network.pth')

# 保存Scaler
package = {
    'scaler_x': scaler_x,
}
joblib.dump(package, 'scaler_x.pkl')


