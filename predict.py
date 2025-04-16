import time
import joblib
import torch
import numpy as np
from utils import BP_NeuralNetwork, data_load
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import matplotlib.pyplot as plt
# ------------------- 加载模型与Scaler ------------------- #
# 加载Scaler
package = joblib.load('scaler_x.pkl')
scaler_x = package['scaler_x']

# 初始化模型结构
model = BP_NeuralNetwork(input_size=8, hidden_size1=10, hidden_size2=10,hidden_size3=10, output_size=1)
model.load_state_dict(torch.load("bp_neural_network.pth", weights_only=True))

# ------------------- 准备新数据 ------------------- #
X, y = data_load("./mat/monte_val_data.mat")
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
target = 0.8
count_pred = np.sum(predictions > target)
count_real = np.sum(y>target)
pass_pred = count_pred/y.size
pass_real = count_real/y.size
print("预测通过率:",pass_pred)
print("实际通过率:",pass_real)

# ------------------- 计算预测损失 ------------------- #
# 计算均方误差（MSE）损失
mse_loss = mean_squared_error(y, predictions)
rmse_loss = root_mean_squared_error(y, predictions)
mean_error = np.mean(np.abs(y-predictions))
max_error = np.max(np.abs(y-predictions))
# ------------------- 打印结果 ------------------- #
print("平均误差（ME）：", mean_error)
print("最大误差（MaxE）：", max_error)
print("均方误差（MSE）：", mse_loss)
print("均方根误差（RMSE）：", rmse_loss)


# ------------------- 绘制预测 vs 真实值回归图 ------------------- #
y_true = y.flatten()        # shape: (N,)
y_pred = predictions.flatten()  # shape: (N,)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号

plt.figure(figsize=(8,6))
plt.scatter(y_true, y_pred, alpha=0.6, color='dodgerblue', label='预测值')

# 理想对角线
plt.plot([0, 1], [0, 1], 'r--', label='理想值 y = x')

# 图形美化
plt.xlabel('真实值', fontsize=12)
plt.ylabel('预测值', fontsize=12)
plt.title('BP神经网络预测效果', fontsize=14)
plt.legend()
plt.grid(True)

# 显示
plt.tight_layout()
plt.show()

# ------------------- 真实值 vs 预测值 直方图对比 ------------------- #
plt.figure(figsize=(8, 6))
plt.hist(y, bins=50, alpha=0.5, label='真实值', color='skyblue', edgecolor='black')
plt.hist(predictions, bins=50, alpha=0.5, label='预测值', color='orange', edgecolor='black')

plt.xlabel('值', fontsize=12)
plt.ylabel('样本数量', fontsize=12)
plt.title('真实值与预测值的分布对比', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
