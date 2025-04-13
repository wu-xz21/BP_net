import time

from utils import BP_NeuralNetwork, torch, np, data_load

# 加载模型
load_time = time.time()
model = BP_NeuralNetwork(input_size=8, hidden_size1=64, hidden_size2=32, output_size=1)
model.load_state_dict(torch.load("bp_neural_network.pth", weights_only=True))

# 假设 new_data 是新的输入数据，形状是 (n_samples, 8)
X = 1


# 将数据转换为 PyTorch 张量
new_data_tensor = torch.tensor(X, dtype=torch.float32)
load_time2 = time.time()

# 使用模型进行预测
model.eval()  # 切换到评估模式
start_time = time.time()
with torch.no_grad():  # 不需要计算梯度
    predictions = model(new_data_tensor)

end_time = time.time()
# 输出预测结果
print(predictions)
print("预测时间：", end_time - start_time, "秒")
print("加载模型时间：", load_time2 - load_time, "秒")