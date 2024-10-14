import time
from dprofile import *

dp = DProfile()


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(0)

def generate_data(num_samples=100):
    X = torch.rand(num_samples, 1) * 10
    y = 2 * X + 1 + torch.randn(num_samples, 1)
    return X, y

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

num_samples = 100
num_epochs = 1000
learning_rate = 0.01


X, y = generate_data(num_samples)
X = X.to('cuda')
y = y.to('cuda')

model = LinearRegressionModel().to('cuda')
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

profile_metric : set = {kDprofile_Metric_DramRead, kDprofile_Metric_DramWrite, kDprofile_Metric_L1Load, kDprofile_Metric_L1Store}

# 训练模型
dp.start_profile(profile_metric)
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        metrics = dp.stop_profile()
        print(metrics)
        dp.start_profile(profile_metric)
