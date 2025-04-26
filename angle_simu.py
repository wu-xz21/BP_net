import torch
from utils import joblib,BP_NeuralNetwork, monte_carlo_pass_fixed_T7T8
import numpy as np
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号
# 加载Scaler
package = joblib.load('scaler_x.pkl')
scaler_x = package['scaler_x']

# 初始化模型结构
model = BP_NeuralNetwork(input_size=8, hidden_size1=10, hidden_size2=10, hidden_size3=10,hidden_size4 = 10, output_size=3)
model.load_state_dict(torch.load("bp_3output_4layer.pth", weights_only=True))
model.eval()  # 切换到评估模式

# ------------------- 准备新数据 ------------------- #
unit_trans_scaler = 0.001*180/np.pi
# x = unit_trans_scaler*np.array([0.2638, 0.05, 0.1771, 0.05, 0.1868, 0.0942])
# x = unit_trans_scaler*np.array([0.5, 0.5, 0.213, 0.052, 0.25, 0.06])    # 利用全部数据的结果
x = unit_trans_scaler*np.array([0.5, 0.5, 0.49, 0.10, 0.6, 0.12])  # 利用全部数据并采用4层神经网络的结果
# x = unit_trans_scaler*np.array([0, 0, 0, 0, 0, 0])
# 搜索范围与步长
T7_range = unit_trans_scaler*np.arange(-1.2, 1.2, 0.05)  #
T8_range = unit_trans_scaler*np.arange(-1.2, 1.2, 0.05)  # 可调整上限
# 初始化记录矩阵
result_grid = np.zeros((len(T7_range), len(T8_range)))

# 记录最优结果
best_T7 = 0
best_T8 = 0
max_sum = -np.inf

# 开始网格搜索
for i, T8 in enumerate(T7_range):
    for j, T7 in enumerate(T8_range):
        result = monte_carlo_pass_fixed_T7T8(x, T7, T8, model, scaler_x, target=0.4)
        result_grid[i, j] = result

        if result == 0:  # 通过！
            if T7 + T8 > max_sum:
                best_T7 = T7
                best_T8 = T8
                max_sum = T7 + T8
                print(f"新最佳: T7={T7:.3f}, T8={T8:.3f}, 总和={max_sum:.3f}")

print(f"\n最终最优：T7={best_T7/unit_trans_scaler:.4f}, T8={best_T8/unit_trans_scaler:.4f}")

# ------------------- 可视化：二值热力图 ------------------- #
plt.figure(figsize=(8,6))
plt.imshow(result_grid,
           extent=[T7_range[0]/unit_trans_scaler, T7_range[-1]/unit_trans_scaler,
                   T8_range[0]/unit_trans_scaler, T8_range[-1]/unit_trans_scaler],
           origin='lower',
           cmap='RdYlGn',  # 红：不通过，绿：通过
           vmin=0, vmax=1)

plt.colorbar(label='Result (0=Pass, 1=Fail)')
plt.xlabel('RX(mrad)')
plt.ylabel('RZ(mrad)')
plt.title('优化后的转角范围')
plt.grid(True)
plt.show()




