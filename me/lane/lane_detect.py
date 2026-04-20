import cv2
import numpy as np


def process_frame(image):
    height, width = image.shape[:2]

    # 1. 预处理：灰阶 + 高斯模糊
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 2. 边缘检测
    canny = cv2.Canny(blur, 50, 150)

    # 3. 提取感兴趣区域 (ROI) - 假设车道在图像下半部的三角形区域
    mask = np.zeros_like(canny)
    polygon = np.array(
        [[(0, height), (width // 2, height // 2 + 50), (width, height)]], np.int32
    )
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(canny, mask)

    # 4. 霍夫变换检测直线
    lines = cv2.HoughLinesP(
        masked_edges, 1, np.pi / 180, threshold=50, minLineLength=40, maxLineGap=100
    )

    # 5. 绘制结果
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return cv2.addWeighted(image, 0.8, line_image, 1, 1)


# 测试代码
print("Starting lane detection...")
cap = cv2.VideoCapture("road_video.mp4")  # 如果有视频文件
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    result = process_frame(frame)
    cv2.imshow("Lane Detection", result)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
