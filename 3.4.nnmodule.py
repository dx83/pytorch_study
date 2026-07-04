import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# nn.Module 클래스를 이용햔 신경망 구현
class SimpleNN_Full(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN_Full, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN_Full(input_size=10, hidden_size=20, output_size=1)
print(model)

for param in model.parameters():
    print(param.shape)



X_train = torch.randn(100, 10)
Y_train = torch.randn(100, 1)
dataset = TensorDataset(X_train, Y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for batch in dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

torch.save(model.state_dict(), 'model.pth')
model = SimpleNN_Full(input_size=10, hidden_size=20, output_size=1)
model.load_state_dict(torch.load('model.pth'))



def check_nan_loss(loss, optimizer):
    if torch.isnan(loss).any():
        print("Loss contains NaN values! Reducing learning rate...")
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        return True
    return False

sample_loss = torch.tensor(float('nan'))
nan_detected = check_nan_loss(sample_loss, optimizer)
print(f"NaN 감지됨: {nan_detected}")



transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

print("데이터 증강 변환이 설정되었습니다.")

