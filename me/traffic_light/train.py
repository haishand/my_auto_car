import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# --- 1. 超參數與環境設定 ---
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./carla_dataset"  # 請修改為你的資料集路徑

# --- 2. 資料預處理 ---
# Carla 資料集圖片較乾淨，通常縮放到 64x64 即可捕捉燈號特徵
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# 載入資料集 (自動根據資料夾名稱分類)
train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
test_data = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# --- 3. 定義 CNN 模型 ---
class TrafficLightCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(TrafficLightCNN, self).__init__()
        # 卷積層 1: 提取顏色邊緣
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # 卷積層 2: 提取形狀特徵
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # 兩次池化後，64x64 -> 32x32 -> 16x16
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = TrafficLightCNN(num_classes=len(train_data.classes)).to(DEVICE)

# --- 4. 定義損失函數與優化器 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# --- 5. 訓練循環 ---
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")


# --- 6. 測試評估 ---
def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    print(f"Classes found: {train_data.classes}")
    train()
    evaluate()
    # 儲存模型
    torch.save(model.state_dict(), "traffic_light_model.pth")
