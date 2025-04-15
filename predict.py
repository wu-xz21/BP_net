import time
import joblib
import torch
import numpy as np
from utils import BP_NeuralNetwork, data_load
from sklearn.metrics import mean_squared_error, max_error, root_mean_squared_error
import matplotlib.pyplot as plt
# ------------------- 加载模型与Scaler ------------------- #
# 加载Scaler
package = joblib.load('scaler_x.pkl')
scaler_x = package['scaler_x']

# 初始化模型结构
model = BP_NeuralNetwork(input_size=8, hidden_size1=10, hidden_size2=10,hidden_size3=10, output_size=1)
model.load_state_dict(torch.load("bp_neural_network.pth", weights_only=True))

# ------------------- 准备新数据 ------------------- #
X, y = data_load("./mat/monte_data.mat")
# X, y = data_load("./mat/hyper_cube_data.mat")

# ⚡ 使用训练阶段的Scaler进行归一化
X_scaled = scaler_x.transform(X)
y = y.reshape(-1,1)
# 转换为 PyTorch 张量
new_data_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# ------------------- 开始预测 ------------------- #
model.eval()  # 切换到评估模式

with torch.no_grad():  # 不需要计算梯度
    predictions_scaled = model(new_data_tensor)

# ------------------- 反归一化输出 ------------------- #
# 转回 numpy
predictions = predictions_scaled.numpy()
# ------------------- 计算预测损失 ------------------- #
# 计算均方误差（MSE）损失
mse_loss = mean_squared_error(y, predictions)
rmse_loss = root_mean_squared_error(y, predictions)
# ------------------- 打印结果 ------------------- #
print("均方误差（MSE）：", mse_loss)
print("均方根误差（RMSE）：", rmse_loss)
# ------------------- 绘制预测 vs 真实值回归图 ------------------- #
y_true = y.flatten()        # shape: (N,)
y_pred = predictions.flatten()  # shape: (N,)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号

plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.6, color='dodgerblue', label='预测值')

# 理想对角线
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='理想值 y = x')

# 图形美化
plt.xlabel('真实值', fontsize=12)
plt.ylabel('预测值', fontsize=12)
plt.title('BP神经网络预测效果', fontsize=14)
plt.legend()
plt.grid(True)

# 显示
plt.tight_layout()
plt.show()

