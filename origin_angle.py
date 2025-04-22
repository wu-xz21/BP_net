import numpy as np
import scipy
import torch
from matplotlib import pyplot as plt

from utils import split_data, BP_NeuralNetwork, train_and_validate, evaluate_model, data_load, joblib
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# 加载Scaler
package = joblib.load('scaler_x.pkl')
scaler_x = package['scaler_x']

# 初始化模型结构
model = BP_NeuralNetwork(input_size=8, hidden_size1=10, hidden_size2=10, hidden_size3=10,hidden_size4 = 10, output_size=1)
model.load_state_dict(torch.load("bp_4layer_alldata.pth", weights_only=True))


# 导入样本数据
mat = './mat/angle_coupling.mat'
mat_data = scipy.io.loadmat(mat)
data = mat_data['data']
unit_trans_scaler = 0.001*180/np.pi
X = data[:,:8]
y = data[:,8]
y = y.reshape(-1,1)

# ⚡ 使用训练阶段的Scaler进行归一化
X_scaled = scaler_x.transform(X)
# 转换为 PyTorch 张量
new_data_tensor = torch.tensor(X_scaled, dtype=torch.float32)
model.eval()  # 切换到评估模式
with torch.no_grad():  # 不需要计算梯度
    predictions_scaled = model(new_data_tensor)
# 转回 numpy
predictions = predictions_scaled.numpy()

# 提取T7, T8 和对应的结果y
T7_values = data[:, 6]  # 第7列，Python索引从0开始
T8_values = data[:, 7]  # 第8列

# T7_pred = unit_trans_scaler*np.linspace(-1.2, 1.2, 30)  # 0 ~ 2，步长0.01
# T8_pred = unit_trans_scaler*np.linspace(-1.2, 1.2, 30)  # 可调整上限


results = y    # y结果
# -------------------- 创建网格 -------------------- #
T7_unique = np.sort(np.unique(T7_values))
T8_unique = np.sort(np.unique(T8_values))

result_grid_true = np.ones((len(T8_unique), len(T7_unique)))  # 默认全1，表示不通过
result_grid_pred = np.ones((len(T8_unique), len(T7_unique)))  # 默认全1，表示不通过
for i, T8 in enumerate(T8_unique):
    for j, T7 in enumerate(T7_unique):
        mask = (np.isclose(T7_values, T7)) & (np.isclose(T8_values, T8))
        matched_results = results[mask]
        matched_predictions = predictions[mask]

        if matched_results.size > 0:
            avg_result = np.mean(matched_results)
            avg_pred = np.mean(matched_predictions)

            # 大于0.4 视为 Pass=0，反之 Fail=1
            result_grid_true[i, j] = 0 if avg_result > 0.4 else 1
            result_grid_pred[i, j] = 0 if avg_pred > 0.4 else 1

fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# 左图：真实结果
im0 = axes[0].imshow(result_grid_true,
                     extent=[T7_unique[0]/unit_trans_scaler, T7_unique[-1]/unit_trans_scaler,
                             T8_unique[0]/unit_trans_scaler, T8_unique[-1]/unit_trans_scaler],
                     origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
axes[0].set_title('Ground Truth Result Grid', fontsize=14)
axes[0].set_xlabel('RX (T7)', fontsize=12)
axes[0].set_ylabel('RZ (T8)', fontsize=12)
axes[0].grid(True)

# 右图：预测结果
im1 = axes[1].imshow(result_grid_pred,
                     extent=[T7_unique[0]/unit_trans_scaler, T7_unique[-1]/unit_trans_scaler,
                             T8_unique[0]/unit_trans_scaler, T8_unique[-1]/unit_trans_scaler],
                     origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
axes[1].set_title('Predicted Result Grid', fontsize=14)
axes[1].set_xlabel('RX (T7)', fontsize=12)
axes[1].set_ylabel('RZ (T8)', fontsize=12)
axes[1].grid(True)

# 单独设置 colorbar
cbar = fig.colorbar(im1, ax=axes, orientation='vertical')
cbar.set_label('Result (0=Pass, 1=Fail)', fontsize=12)

plt.show()