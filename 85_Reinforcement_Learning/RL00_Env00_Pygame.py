
### Pygame #####################################################################

import pygame
import sys

# 1. Pygame 초기화
pygame.init()

# 2. 화면 생성
screen = pygame.display.set_mode((800, 600))  # 창 크기: 가로 800, 세로 600
pygame.display.set_caption("My Pygame Window")  # 창 제목

# 3. 게임 루프
running = True
while running:
    # 이벤트 처리
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # 창 닫기 버튼
            running = False

        # 특정 키 입력 처리
        if event.type == pygame.KEYDOWN:  # 키가 눌렸을 때
            if event.key == pygame.K_ESCAPE:  # ESC 키 확인
                print("ESC key pressed! Exiting the game...")
                running = False  # 게임 루프 종료

    # 화면 색상 채우기 (RGB 값)
    screen.fill((0, 0, 0))  # 검정색으로 채움

    # 화면 업데이트
    pygame.display.update()

# 4. Pygame 종료
pygame.quit()
sys.exit()


