import pygame
import random
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# ====================== 參數設定 ======================
WIDTH, HEIGHT = 800, 600
CAR_SIZE = 8
SPEED = 4
ROT_SPEED = 7
FPS = 60
TRACK_WIDTH = 120

# DQN 超參數
GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

WHITE, BLACK, GRAY, GREEN, RED = (
    (255, 255, 255),
    (0, 0, 0),
    (128, 128, 128),
    (0, 255, 0),
    (255, 0, 0),
)


# ====================== 神經網路 ======================
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, action_size),
        )

    def forward(self, x):
        return self.fc(x)


# ====================== 賽道邏輯 ======================
def get_track_center(y):
    # 彎道中心計算
    return WIDTH // 2 + int(math.sin(y * 0.01) * 150)


# ====================== 小車與 AI ======================
class Car:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y = HEIGHT - 50
        self.x = get_track_center(self.y)
        self.angle = -90

    def move(self, action):
        if action == 1:
            self.angle -= ROT_SPEED  # 左
        elif action == 2:
            self.angle += ROT_SPEED  # 右
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * SPEED
        self.y += math.sin(rad) * SPEED

    def get_state(self):
        # 狀態：[水平偏移量, 角度偏移] (歸一化處理)
        center = get_track_center(self.y)
        return np.array(
            [(self.x - center) / 100.0, (self.angle + 90) / 45.0], dtype=np.float32
        )


# ====================== 主程式 ======================
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    # 初始化 DQN
    state_size = 2
    action_size = 3
    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    memory = deque(maxlen=MEMORY_SIZE)

    epsilon = 1.0
    episode = 0
    car = Car()

    while True:
        screen.fill(BLACK)
        for ty in range(0, HEIGHT, 20):
            tc = get_track_center(ty)
            pygame.draw.rect(screen, GRAY, (tc - TRACK_WIDTH // 2, ty, TRACK_WIDTH, 20))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # 1. 選擇動作 (Epsilon-greedy)
        state = car.get_state()
        state_t = torch.FloatTensor(state).unsqueeze(0)
        if random.random() < epsilon:
            action = random.randint(0, 2)
        else:
            with torch.no_grad():
                action = policy_net(state_t).argmax().item()

        # 2. 執行並觀察
        car.move(action)
        center = get_track_center(car.y)
        out = abs(car.x - center) > (TRACK_WIDTH // 2)
        reward = 1.0 if not out else -10.0
        done = out or car.y < 0 or car.y > HEIGHT
        next_state = car.get_state()

        # 3. 儲存經驗
        memory.append((state, action, reward, next_state, done))

        if done:
            car.reset()
            episode += 1
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        # 4. 經驗回放訓練
        if len(memory) > BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            b_s, b_a, b_r, b_ns, b_d = zip(*batch)

            b_s = torch.FloatTensor(np.array(b_s))
            b_a = torch.LongTensor(b_a).unsqueeze(1)
            b_r = torch.FloatTensor(b_r)
            b_ns = torch.FloatTensor(np.array(b_ns))
            b_d = torch.FloatTensor(b_d)

            q_values = policy_net(b_s).gather(1, b_a)
            next_q_values = target_net(b_ns).max(1)[0].detach()
            expected_q = b_r + (GAMMA * next_q_values * (1 - b_d))

            loss = nn.MSELoss()(q_values.squeeze(), expected_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 每 10 個 episode 更新一次目標網路
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 繪圖
        pygame.draw.circle(
            screen, GREEN if not out else RED, (int(car.x), int(car.y)), CAR_SIZE
        )
        info = font.render(
            f"Epi: {episode} | Eps: {epsilon:.2f} | Mem: {len(memory)}", True, WHITE
        )
        screen.blit(info, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
