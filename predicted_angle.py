import numpy as np
import torch
from matplotlib import pyplot as plt
from utils import BP_NeuralNetwork, joblib

# 加载Scaler
package = joblib.load('scaler_x.pkl')
scaler_x = package['scaler_x']

# 初始化模型结构
model = BP_NeuralNetwork(input_size=8, hidden_size1=10, hidden_size2=10, hidden_size3=10,hidden_size4 = 10, output_size=1)
model.load_state_dict(torch.load("bp_4layer_alldata.pth", weights_only=True))

# T7、T8网格自定义
unit_trans_scaler = 0.001*180/np.pi
T7_pred = unit_trans_scaler * np.linspace(-1.2, 1.2, num=30)  # T7网格，50个点
T8_pred = unit_trans_scaler * np.linspace(-1.2, 1.2, num=30)  # T8网格，50个点

# 生成网格
T7_mesh, T8_mesh = np.meshgrid(T7_pred, T8_pred)
T7_flat = T7_mesh.ravel()
T8_flat = T8_mesh.ravel()

# 其他6个输入设为0
T_other = np.zeros((len(T7_flat), 6))  # shape = [点数, 6]

# 拼接完整输入：T1~T6为0，T7、T8为网格值
X_test = np.column_stack([T_other, T7_flat, T8_flat])  # shape: [点数, 8]

# 用训练时Scaler归一化
X_test_scaled = scaler_x.transform(X_test)

# 转为Tensor，预测
X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor).numpy()

# 结果转换为Pass/Fail
result_grid = (y_pred.reshape(T8_mesh.shape) <= 0.4).astype(int)  # <=0.4 = Fail(1), 否则 Pass(0)

# ------------------- 可视化 ------------------- #
plt.figure(figsize=(8,6))
plt.imshow(result_grid,
           extent=[T7_pred[0]/unit_trans_scaler, T7_pred[-1]/unit_trans_scaler,
                   T8_pred[0]/unit_trans_scaler, T8_pred[-1]/unit_trans_scaler],
           origin='lower', cmap='RdYlGn', vmin=0, vmax=1)

plt.colorbar(label='Prediction (0=Pass, 1=Fail)')
plt.xlabel('RX (T7)')
plt.ylabel('RZ (T8)')
plt.title('Neural Network Predicted Pass/Fail Map')
plt.grid(True)
plt.show()
