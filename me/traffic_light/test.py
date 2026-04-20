import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


# 1. 必須定義與訓練時完全相同的模型結構
class TrafficLightCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(TrafficLightCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
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


# 2. 設定環境與類別標籤 (順序需與訓練時的 ImageFolder 一致)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["back", "green", "red", "yellow"]  # 請確保順序與資料夾字母排序一致

# 3. 載入模型
model = TrafficLightCNN(num_classes=len(CLASSES))
model.load_state_dict(torch.load("traffic_light_model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()  # 切換到評估模式（關閉 Dropout）

# 4. 定義預處理 (必須與訓練時一致)
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict(image_path):
    # 讀取並轉換圖片
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # 增加 Batch 維度

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)  # 轉為百分比
        confidence, predicted = torch.max(probabilities, 1)

    label = CLASSES[predicted.item()]
    score = confidence.item()

    print(f"預測結果: {label} (信心度: {score:.2%})")
    return label, score


# 5. 執行測試
if __name__ == "__main__":
    test_img = "test_light.png"  # 放入你的圖片路徑
    predict(test_img)

    test_img2 = "test_light2.jpg"  # 放入你的圖片路徑
    predict(test_img2)
