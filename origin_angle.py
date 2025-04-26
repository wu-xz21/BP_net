import numpy as np
import scipy
import torch
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from utils import BP_NeuralNetwork, joblib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文字体
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 加载Scaler
package = joblib.load('scaler_x.pkl')
scaler_x = package['scaler_x']

# 初始化模型结构
model = BP_NeuralNetwork(input_size=8, hidden_size1=10, hidden_size2=10, hidden_size3=10, hidden_size4=10, output_size=3)
model.load_state_dict(torch.load("bp_3output_4layer.pth", weights_only=True))

# 导入样本数据
mat = './mat/3output_simu_angle.mat'
mat_data = scipy.io.loadmat(mat)
data = mat_data['data']
unit_trans_scaler = 0.001 * 180 / np.pi
X = data[:, :8]
y = data[:, 8:11]

# 使用训练阶段的Scaler进行归一化
X_scaled = scaler_x.transform(X)
new_data_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# 预测
model.eval()
with torch.no_grad():
    predictions_scaled = model(new_data_tensor)
predictions = predictions_scaled.numpy()

# 提取 RX (T7) 和 RZ (T8)
T7_values = data[:, 6]
T8_values = data[:, 7]

# 获取唯一角度值
T7_unique = np.sort(np.unique(T7_values))
T8_unique = np.sort(np.unique(T8_values))

# 初始化 3 通道的结果矩阵
result_grid_true = np.ones((3, len(T8_unique), len(T7_unique)))
result_grid_pred = np.ones((3, len(T8_unique), len(T7_unique)))

for ch in range(3):
    for i, T8 in enumerate(T8_unique):
        for j, T7 in enumerate(T7_unique):
            mask = (np.isclose(T7_values, T7)) & (np.isclose(T8_values, T8))
            matched_results = y[mask][:, ch]
            matched_predictions = predictions[mask][:, ch]

            if matched_results.size > 0:
                avg_result = np.mean(matched_results)
                avg_pred = np.mean(matched_predictions)

                result_grid_true[ch, i, j] = 0 if avg_result > 0.4 else 1
                result_grid_pred[ch, i, j] = 0 if avg_pred > 0.4 else 1

# -------------------- 绘制实际值 -------------------- #
fig_true, axes_true = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
fig_true.suptitle('Zemax仿真结果', fontsize=18)

for ch in range(3):
    im = axes_true[ch].imshow(result_grid_true[ch],
                              extent=[T7_unique[0]/unit_trans_scaler, T7_unique[-1]/unit_trans_scaler,
                                      T8_unique[0]/unit_trans_scaler, T8_unique[-1]/unit_trans_scaler],
                              origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
    axes_true[ch].set_title(f'通道{ch + 1}', fontsize=14)
    axes_true[ch].set_xlabel('RX (T7)')
    axes_true[ch].set_ylabel('RZ (T8)')
    axes_true[ch].grid(True)

fig_true.colorbar(im, ax=axes_true, orientation='vertical', label='结果 (0=通过, 1=不通过)')

# -------------------- 绘制预测值 -------------------- #
fig_pred, axes_pred = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
fig_pred.suptitle('BP神经网络预测结果', fontsize=18)

for ch in range(3):
    im = axes_pred[ch].imshow(result_grid_pred[ch],
                              extent=[T7_unique[0]/unit_trans_scaler, T7_unique[-1]/unit_trans_scaler,
                                      T8_unique[0]/unit_trans_scaler, T8_unique[-1]/unit_trans_scaler],
                              origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
    axes_pred[ch].set_title(f'通道{ch + 1}', fontsize=14)
    axes_pred[ch].set_xlabel('RX (T7)')
    axes_pred[ch].set_ylabel('RZ (T8)')
    axes_pred[ch].grid(True)

fig_pred.colorbar(im, ax=axes_pred, orientation='vertical', label='结果 (0=通过, 1=不通过)')

plt.show()

# 画出相应的混淆矩阵热图
# 通道数
num_channels = 3
channel_titles = ["通道1", "通道2", "通道3"]

# 创建 Figure
fig, axes = plt.subplots(1, num_channels, figsize=(15, 5))
fig.suptitle("每个通道的混淆矩阵热图", fontsize=16)

for i in range(num_channels):
    # 拉平成一维
    y_true_flat = result_grid_true[:, :, i].flatten()
    y_pred_flat = result_grid_pred[:, :, i].flatten()

    # 计算混淆矩阵
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=[0, 1])

    # 绘制热图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['预测 Pass', '预测 Fail'],
                yticklabels=['真实 Pass', '真实 Fail'],
                ax=axes[i])
    axes[i].set_title(channel_titles[i])
    axes[i].set_xlabel('预测标签')
    axes[i].set_ylabel('真实标签')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# 打印出每个通道的准确率
accuracies = []

for i in range(num_channels):
    y_true_flat = result_grid_true[:, :, i].flatten()
    y_pred_flat = result_grid_pred[:, :, i].flatten()

    acc = accuracy_score(y_true_flat, y_pred_flat)
    accuracies.append(acc)
    print(f"通道{i + 1} 的准确率: {acc:.4f}")