import numpy as np

example = False

class CustomGridWorld:
    def __init__(self, grid_map=None, grid_size=4, start=(0, 0), goal=None, obstacles=None, traps=None, treasures=None, transition_prob=1.0,
                reward_step=0, reward_goal=10, reward_trap=-2, reward_obstacle=-1, reward_treasure=1, random_state=None):
        """
        action : 0 상, 1 하, 2 좌, 3 우
        self.step(action) : action에 대한 실행
        """
        self.grid_size = grid_size
        self.start = start
        self.goal = (grid_size-1, grid_size-1) if goal is None else goal
        self.obstacles = obstacles
        self.traps = traps
        self.treasures = treasures

        self.nS = self.grid_size * self.grid_size
        self.nA = 4

        self.grid_map = None
        self.render_map = None

        # Transition probability (확률적으로 다른 방향으로 움직임)
        self.transition_prob = transition_prob  # 기본적으로 1.0 = 100% 정해진 방향으로 이동

        self.reward_step = reward_step
        self.reward_goal = reward_goal
        self.reward_trap = reward_trap
        self.reward_obstacle = reward_obstacle
        self.reward_treasure = reward_treasure

        self.agent_label = 1
        self.goal_label = 9
        self.obstacle_label = -1
        self.trap_label = -2
        self.treasure_label = 2

        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        self.cur_info = ([0,0], [0,0], 0, False)
        self.init_info = {}
        self.initialize(grid_map)

    def set_rewards(self, reward_step=0, reward_goal=10, reward_trap=-2, reward_obstacle=-1, reward_treasure=1):
        self.reward_step = reward_step
        self.reward_goal = reward_goal
        self.reward_trap = reward_trap
        self.reward_obstacle = reward_obstacle
        self.reward_treasure = reward_treasure 

    def initialize(self, grid_map=None):
        """ grid_map이 제공되면 해당 맵을 사용, 제공되지 않으면 랜덤 맵 생성 """
        if grid_map is not None:
            self.grid_size = grid_map.shape[0]
            self.grid_map = np.array(grid_map)
            self.grid_size = self.grid_map.shape[0]  # 입력된 맵의 크기 기반으로 설정

            goal = np.where(np.array(custom_map)==9)
            self.goal = [tuple(arr) for arr in np.stack(goal).T][0]

            obstacles = np.where(np.array(custom_map)==-2)
            self.obstacles = [tuple(arr) for arr in np.stack(obstacles).T]

            traps = np.where(np.array(custom_map)==-1)
            self.traps = [tuple(arr) for arr in np.stack(traps).T]
            
            treasures = np.where(np.array(custom_map)==2)
            self.treasures = [tuple(arr) for arr in np.stack(treasures).T]

        else:
            self.grid_map = np.zeros((self.grid_size, self.grid_size), dtype=int)

            # 목표 설정
            self.grid_map[self.goal] = self.goal_label

            # obstacles을 지정하지 않았을 때 random 배치
            if self.obstacles is None:
                self.obstacles = []
                self.set_random_obstacles()
            else:
                for obstacle in self.obstacles:
                    self.grid_map[obstacle] = self.obstacle_label  # 장애물은 -1로 표시

            # traps을 지정하지 않았을 때 random 배치
            if self.traps is None:
                self.traps = []
                self.set_random_traps()
            else:
                for trap in self.traps:
                    self.grid_map[trap] = self.trap_label  # 함정은 -2로 표시

            # 중간 보상 지점 설정
            if self.treasures is None:
                self.treasures = []
                self.set_random_treasures()
            else:
                for treasure in self.treasures:
                    self.grid_map[treasure] = 2  # 중간보상(보물)은 2로 표시

        self.nS = self.grid_size * self.grid_size

        self.init_info['obstacles'] = self.obstacles.copy()
        self.init_info['traps'] = self.traps.copy()
        self.init_info['treasures'] = self.treasures.copy()

        self.cur_state = tuple(list(self.start))
        self.render_map = self.render(verbose=0, return_map=True)

    def set_random_obstacles(self):
        """ 랜덤으로 장애물을 배치 """
        print('create random obstacle')
        num_obstacles = self.rng.randint(1, int(self.grid_size*self.grid_size*0.2))  # 랜덤 장애물 개수

        # 랜덤 장애물 설정
        for _ in range(num_obstacles):
            obstacle_pos = (self.rng.randint(0, self.grid_size), self.rng.randint(0, self.grid_size))
            if obstacle_pos != self.goal and obstacle_pos != self.start:
                self.grid_map[obstacle_pos] = self.obstacle_label
                self.obstacles.append(obstacle_pos)
        self.obstacles = list(set(self.obstacles))      # 중복제거

    def set_random_traps(self):
        """ 랜덤으로 함정을 배치 """
        print('create random trap')
        num_traps = self.rng.randint(1, int(self.grid_size*self.grid_size*0.2))  # 랜덤 함정 개수

        # 랜덤 함정 설정
        for _ in range(num_traps):
            trap_pos = (self.rng.randint(0, self.grid_size), self.rng.randint(0, self.grid_size))
            if trap_pos != self.goal and trap_pos != self.start and trap_pos not in self.obstacles:
                self.grid_map[trap_pos] = self.trap_label
                self.traps.append(trap_pos)
        self.traps = list(set(self.traps))      # 중복제거

    def set_random_treasures(self):
        """ 랜덤으로 중간보상(보물)을 배치 """
        print('create random treasures')
        num_treasures = self.rng.randint(1, int(self.grid_size*self.grid_size*0.2))  # 랜덤 중간보상(보물) 개수

        # 랜덤 중간보상(보물) 설정
        for _ in range(num_treasures):
            treasure_pos = (self.rng.randint(0, self.grid_size), self.rng.randint(0, self.grid_size))
            if treasure_pos != self.goal and treasure_pos != self.start and treasure_pos not in self.obstacles and treasure_pos not in self.traps:
                self.grid_map[treasure_pos] = 2
                self.treasures.append(treasure_pos)
        self.treasures = list(set(self.treasures))      # 중복제거

    def reset(self):
        """ 환경을 초기 상태로 리셋 """
        self.obstacles = self.init_info['obstacles']
        self.traps = self.init_info['traps']
        self.treasures = self.init_info['treasures']
        self.initialize()
        return tuple(self.cur_state)

    def step(self, action=None, verbose=0):
        """ 에이전트가 이동한 후 상태, 보상, 완료 여부 반환 """

        if action is None:
            action = np.random.choice([0, 1, 2, 3])

        # Transition Probability 적용: action과 다른 방향으로 이동할 확률이 발생
        actual_action = self.apply_transition_probability(action)

        next_state = list(self.cur_state)  # 새로운 위치 계산
        if actual_action == 0:  # 상
            next_state[0] = max(0, self.cur_state[0] - 1)
        elif actual_action == 1:  # 하
            next_state[0] = min(self.grid_size - 1, self.cur_state[0] + 1)
        elif actual_action == 2:  # 좌
            next_state[1] = max(0, self.cur_state[1] - 1)
        elif actual_action == 3:  # 우
            next_state[1] = min(self.grid_size - 1, self.cur_state[1] + 1)

        # 보상 계산
        done = False
        
        # 장애물에 부딪히면 움직이지 않음
        if tuple(next_state) in self.obstacles:
            reward = self.reward_obstacle
            next_state = self.cur_state  # 이동을 취소하고 원래 위치로 돌아감
        elif tuple(next_state) == self.goal:
            reward = self.reward_goal  # 목표 도달 시 보상
            done = True  # 목표 도달 시 종료
        elif tuple(next_state) in self.traps:
            reward = self.reward_trap  # 함정에 빠질 때 보상
            next_state = self.reset()  # 함정에 빠지면 리셋 후 진행
        elif tuple(next_state) in self.treasures:
            reward = self.reward_treasure   # 보물 지점에서 보상
            self.treasures.remove(tuple(next_state))  # 보물을 먹은 후 제거
            self.grid_map[tuple(next_state)] = 0
        else:
            reward = self.reward_step
        
        if verbose > 0:
            if verbose == 2:
                self.render(verbose=1, return_map=False)
            print(next_state, reward, done)
        else:
            self.render(verbose=False, return_map=False)
        
        self.cur_info = (tuple(self.cur_state), tuple(next_state), reward, done)
        self.cur_state = tuple(next_state)

        return self.cur_info

    def apply_transition_probability(self, action):
        """ transition probability에 따라 실제 이동 액션을 결정 """
        if self.rng.random() < self.transition_prob:
            # 원래 액션대로 이동
            return action
        else:
            # 다른 액션으로 이동
            possible_actions = [0, 1, 2, 3]
            possible_actions.remove(action)  # 현재 액션 제외
            return self.rng.choice(possible_actions)

    def render(self, verbose=1, return_map=False):
        """ 현재 그리드를 출력 """
        self.render_map = self.grid_map.copy()  # 맵 복사
        self.render_map[tuple(self.cur_state)] = self.agent_label  # 에이전트 위치 표시

        if verbose:
            print(self.render_map)
            print()
        
        if return_map:
            return self.render_map



if example:
    # 환경 사용 예시
    # 사용자 정의 장애물과 함정
    traps = [(1, 1), (2, 3)]
    obstacles = [(1, 2), (2, 1)]
    treasures = [(0,3), (3,0)]

    # env = CustomGridWorld()
    env = CustomGridWorld(grid_size=5)
    # env = CustomGridWorld(grid_size=5, traps=[], obstacles=[], treasures=[])
    # env = CustomGridWorld(grid_size=5, obstacles=obstacles)
    # env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps)
    # env = CustomGridWorld(grid_size=5, obstacles=obstacles, traps=traps, treasures=treasures)

    # custom_map = [
    #     [0, 0, 0, 1],
    #     [0, -1, 0, 0],
    #     [0, 1, 0, 0],
    #     [0, 0, 0, 9]  # 9는 목표 위치
    # ]
    # env.initialize(custom_map)

    env.render()

    env.reset()
    env.traps
    env.obstacles
    env.treasures
    
    # env.init_info
    

    # 간단한 에피소드 실행
    state = env.reset()
    env.render()
    done = False

    t = 0
    while not done:
        action = np.random.choice([0, 1, 2, 3])  # 임의의 액션 선택 (상, 하, 좌, 우)
        next_state, reward, done = env.step(action)
        print(f"Action: {action}, Next State: {next_state}, Reward: {reward}")
        env.render()

        if t > 10000:
            break
        t+=1
    done

