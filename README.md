# 📘 项目名称：基于BP神经网络的公差设计与预测分析工具

## 💡 项目简介

本项目基于PyTorch框架，实现了一个针对机械结构或工艺特征数据的**公差预测与优化分析工具**。通过训练BP神经网络模型，完成输入特征到目标性能指标的拟合，并结合蒙特卡洛采样与PSO粒子群优化，辅助完成复杂结构的公差反向设计，降低测试和制造风险。

---

## 📂 项目结构

```
.
├── train.py               # 脚本1：训练模型并保存
├── predict.py             # 脚本2：加载模型并进行预测分析，输出误差与可视化
├── utils.py               # 脚本3：核心方法，包含模型类、训练流程、数据划分与优化算法
├── tolerance_optimize.py  # 脚本4：调用PSO公差设计，输出最优解
├── /mat                   # 存储.mat样本数据文件的目录
├── bp_neural_network.pth  # 训练好的模型权重文件
├── scaler_x.pkl           # 训练阶段特征标准化的Scaler对象
```

---

## ⚙️ 环境依赖

请确保以下Python库已正确安装：

```bash
torch >= 1.12
scikit-learn >= 1.2
matplotlib >= 3.6
numpy >= 1.22
scipy >= 1.10
scikit-opt >= 0.5.0  # PSO优化算法库
joblib >= 1.2
```

推荐使用 `conda` 或 `pip` 创建虚拟环境。

---

## 🚀 脚本说明

### 1️⃣ `train.py` - 模型训练

- 功能：
    - 从 `mat/hyper_cube_data.mat` 加载训练数据；
    - 完成特征标准化；
    - 构建三层BP神经网络；
    - 支持早停的训练过程；
    - 保存训练好的模型权重 `bp_neural_network.pth` 和 `scaler_x.pkl`。

---

### 2️⃣ `predict.py` - 模型预测与评估

- 功能：
    - 加载已训练好的模型与Scaler；
    - 读取新数据文件（如 `monte_val_data.mat`）；
    - 输出预测误差（MSE、RMSE、平均误差等）；
    - 自动绘制两种图表：
        - 真实值 vs 预测值散点回归图；
        - 真实值与预测值直方图对比。

---

### 3️⃣ `utils.py` - 方法库

- 核心组件：
    - `BP_NeuralNetwork`：定义三层隐藏层的全连接神经网络；
    - `train_and_validate`：训练过程封装，支持早停；
    - `evaluate_model`：标准误差评估；
    - `split_data`：训练集、验证集、测试集划分；
    - `data_load`：MATLAB格式数据解析；
    - `tolerance`：集成PSO粒子群算法的公差反向优化；
    - `monte_carlo_pass`：蒙特卡洛约束条件计算。

---

### 4️⃣ `tolerance_optimize.py` - 公差反向优化

- 功能：
    - 加载已训练的神经网络；
    - 调用 `tolerance()` 函数进行公差反向设计；
    - 自动执行蒙特卡洛分析约束，得到满足条件的最优公差范围；
    - 输出PSO搜索的损失收敛曲线与最优解。

---

## 🏁 运行示例

1️⃣ **训练模型**

```bash
python train.py
```

2️⃣ **预测新数据集效果**

```bash
python predict.py
```

3️⃣ **公差设计优化**

```bash
python tolerance_optimize.py
```

---

## 💡 注意事项

- 所有输入数据必须为MATLAB `.mat` 文件，且矩阵格式要求：
    - 数据矩阵 `data`：
        - 前8列：输入特征；
        - 第9列：目标标签。

- 模型输出已通过`Sigmoid`归一化，建议目标标签范围应在 [0,1]。

- 若进行公差优化，建议合理设置`optimize_T7`、`optimize_T8`，防止PSO迭代时出现无穷大解。

---

## ✍️ 作者

Sage_gooooo 

---
