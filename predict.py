import time
import joblib
import torch
import numpy as np
from utils import BP_NeuralNetwork, data_load
from sklearn.metrics import mean_squared_error
# ------------------- 加载模型与Scaler ------------------- #
# 加载Scaler
package = joblib.load('scaler_x.pkl')
scaler_x = package['scaler_x']

# 初始化模型结构
model = BP_NeuralNetwork(input_size=8, hidden_size1=64, hidden_size2=32, output_size=1)
model.load_state_dict(torch.load("bp_neural_network.pth", weights_only=True))

# ------------------- 准备新数据 ------------------- #
# 假设 new_data 是新的输入，形状: (n_samples, 8)
# 示例：这里输入1行数据
X = np.array([
    [0.0285408257098325, 0.00731116002027779, 0.00863269309864845, 0.00113685991337620, -0.0332615453836888, -0.00510556734175169, -0.0208632351109377, 0.0274781074933314]
])
X, y = data_load("./mat/monte_data.mat")

# ⚡ 使用训练阶段的Scaler进行归一化
X_scaled = scaler_x.transform(X)

# 转换为 PyTorch 张量
new_data_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# ------------------- 开始预测 ------------------- #
model.eval()  # 切换到评估模式
start_time = time.time()

with torch.no_grad():  # 不需要计算梯度
    predictions_scaled = model(new_data_tensor)

end_time = time.time()

# ------------------- 反归一化输出 ------------------- #
# 转回 numpy
predictions = predictions_scaled.numpy()
# ------------------- 计算预测损失 ------------------- #
# 计算均方误差（MSE）损失
mse_loss = mean_squared_error(y, predictions)

# ------------------- 打印结果 ------------------- #
# print("预测结果：", predictions)
print("均方误差（MSE）：", mse_loss)
print("预测耗时：", end_time - start_time, "秒")
