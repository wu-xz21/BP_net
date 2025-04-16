import torch
from utils import tolerance,joblib,BP_NeuralNetwork

# 加载Scaler
package = joblib.load('scaler_x.pkl')
scaler_x = package['scaler_x']

# 初始化模型结构
model = BP_NeuralNetwork(input_size=8, hidden_size1=10, hidden_size2=10, hidden_size3=10, output_size=1)
model.load_state_dict(torch.load("bp_neural_network.pth", weights_only=True))
model.eval()  # 切换到评估模式

# ------------------- 准备新数据 ------------------- #
X, y = tolerance(model, scaler_x,0.4,0.4)




