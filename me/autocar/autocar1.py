import pygame
import random
import numpy as np
import math

# 在 import 之後定義顏色
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


# ====================== 參數 ======================
WIDTH, HEIGHT = 800, 600
CAR_SIZE = 8
SPEED = 4
ROT_SPEED = 7
FPS = 60

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1
TRACK_WIDTH = 120  # 賽道寬度


class Car:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y = HEIGHT - 50  # 起點在畫面底部上方一點
        # 動態計算當前高度對應的賽道中心
        self.x = get_track_center(self.y)
        self.angle = -90  # 朝向正上方

    def move(self, action):
        if action == 1:
            self.angle -= ROT_SPEED  # 左轉
        elif action == 2:
            self.angle += ROT_SPEED  # 右轉

        rad = math.radians(self.angle)
        self.x += math.cos(rad) * SPEED
        self.y += math.sin(rad) * SPEED

    def get_state(self, track_center_x):
        # 狀態：相對於賽道中心的偏移量 (-100~100 映射到 0~19)
        offset = (self.x - track_center_x) // 10
        s1 = int(np.clip(offset + 10, 0, 19))
        # 狀態：角度偏移 (-45~45度 映射到 0~19)
        angle_diff = (self.angle + 90) // 5
        s2 = int(np.clip(angle_diff + 10, 0, 19))
        return s1, s2


def get_track_center(y):
    # 用正弦波產生彎道：中心點隨 y 座標擺動
    return WIDTH // 2 + int(math.sin(y * 0.01) * 150)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    car = Car()
    q_table = np.zeros((20, 20, 3))
    episode = 0

    while True:
        screen.fill(BLACK)

        # 繪製彎道（簡單繪製多個點組成賽道）
        for ty in range(0, HEIGHT, 20):
            tc = get_track_center(ty)
            pygame.draw.rect(screen, GRAY, (tc - TRACK_WIDTH // 2, ty, TRACK_WIDTH, 20))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        # 獲取當前高度的賽道中心
        current_center = get_track_center(car.y)
        s1, s2 = car.get_state(current_center)

        # AI 選擇動作
        action = (
            random.randint(0, 2)
            if random.uniform(0, 1) < EPSILON
            else np.argmax(q_table[s1][s2])
        )
        car.move(action)

        # 判定撞牆
        out = abs(car.x - current_center) > (TRACK_WIDTH // 2)
        reward = 1 if not out else -200

        # 學習
        next_center = get_track_center(car.y)  # 移動後的新中心
        ns1, ns2 = car.get_state(next_center)
        q_table[s1][s2][action] += LEARNING_RATE * (
            reward
            + DISCOUNT_FACTOR * np.max(q_table[ns1][ns2])
            - q_table[s1][s2][action]
        )

        if out or car.y < 0 or car.y > HEIGHT:
            car.reset()
            episode += 1

        # 畫車
        pygame.draw.circle(
            screen, GREEN if not out else RED, (int(car.x), int(car.y)), CAR_SIZE
        )

        msg = font.render(f"Episode: {episode} | AI is learning curves...", True, WHITE)
        screen.blit(msg, (10, 10))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
