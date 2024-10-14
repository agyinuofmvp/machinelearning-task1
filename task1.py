import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#加载波士顿房价数据集
train_data = pd.read_csv("/mnt/d/train.csv")
test_data = pd.read_csv("/mnt/d/test.csv")

#检查缺失值
print(train_data.isnull().sum())

#选择特征变量和目标变量
x_train = train_data.drop(columns=['ID', 'medv'])
y_train = train_data['medv'].values
x_test = test_data.drop(columns=['ID'])

#数据归一化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#转化为PyTorch的Tensor
x_train = torch.tensor(x_train, dtype=torch.float32)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

#划分训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

#构建神经网络模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(x_train.shape[1], 64) #输入层
        self.fc2 = nn.Linear(64, 64) #隐藏层
        self.fc3 = nn.Linear(64, 1) #输出层
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        y_pred = self.fc3(x)
        return y_pred
model = LinearModel()

#损失函数及优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#训练模型
train_losses = []
val_losses = []
for epoch in range(1000):
    model.train()  #设置为训练模式
    optimizer.zero_grad()
    y_pred = model(x_train)
    train_loss = criterion(y_pred, y_train)
    train_loss.backward()
    optimizer.step()
    
    model.eval()  #测试模式
    with torch.no_grad():
        y_val_pred = model(x_val)
        val_loss = criterion(y_val_pred, y_val)
    
    #将损失转化为数值并保存
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{1000}],Train Loss: {train_loss.item():.4f}, Val Loss:{val_loss.item():.4f}')
    
#可视化损失曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.show()
plt.savefig('task1.png')

#划分房价：低--（0，15）；中--（15，30）；高--（30+）
bins = [0, 15, 30, np.inf]
labels = [1, 2, 3]
y_val_bin = pd.cut(y_val.numpy().squeeze(), bins=bins, labels=labels, right=False)
y_val_pred_bin = pd.cut(y_val_pred.numpy().squeeze(), bins=bins, labels=labels, right=False)

#计算并打印每个区间的准确率
accuracy_per_bin = {}
for label in labels:
    correct = (y_val_bin == label) & (y_val_pred_bin == label)
    accuracy_per_bin[label] = correct.sum() / (y_val_bin == label).sum()
for label, accuracy in accuracy_per_bin.items():
    print(f'Price Range {bins[label - 1]}-{bins[label]}: Accuracy = {accuracy:.2f}')
    
#评估模型性能
rmse = np.sqrt(val_loss.item())
print(f'Test RMSE: {rmse:.4f}')
    
#打印预测房价
model.eval()  #测试模式
with torch.no_grad():
    pred = model(x_test)
predicted_price = pred.detach().numpy() 
print("Predicted prices:", predicted_price)   

