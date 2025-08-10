import minari
import json

# 저장할 파일 이름
file_name = "minari_datasets_list.txt"

print("Minari 원격 데이터셋 목록을 가져오는 중...")

try:
    # 1. Minari의 전체 원격 데이터셋 목록을 딕셔너리 형태로 가져옵니다.
    remote_datasets = minari.list_remote_datasets()
    print("목록을 성공적으로 가져왔습니다.")

    # 2. 가져온 목록을 보기 좋게 정렬하여 txt 파일로 저장합니다.
    with open(file_name, "w", encoding="utf-8") as f:
        # json.dump를 사용하면 딕셔너리를 깔끔하게 저장할 수 있습니다.
        json.dump(remote_datasets, f, indent=4, ensure_ascii=False)
    print(f"전체 목록을 '{file_name}' 파일에 저장했습니다.")

    # 3. 데이터셋 이름(key)들 중에서 'hopper'가 포함된 것이 있는지 찾습니다.
    found_hopper_datasets = []
    for name in remote_datasets.keys():
        if 'hopper' in name:
            found_hopper_datasets.append(name)

    # 4. 최종 결과를 출력합니다.
    print("\n--- 'hopper' 데이터셋 검색 결과 ---")
    if found_hopper_datasets:
        print(f"✅ 총 {len(found_hopper_datasets)}개의 'hopper' 데이터셋을 찾았습니다!")
        for dataset_name in found_hopper_datasets:
            print(f"  -> {dataset_name}")
    else:
        print("❌ 'hopper'가 포함된 데이터셋을 원격 목록에서 찾지 못했습니다.")
        print("   (네트워크 문제 또는 라이브러리 버전 문제를 다시 확인해 보세요.)")

except Exception as e:
    print(f"\n스크립트 실행 중 오류가 발생했습니다.")
    print(f"에러: {e}")
    

    
import minari
import numpy as np

# Load the local dataset
dataset = minari.load_dataset("mujoco/hopper/expert-v0", download=True)

print("--- ✅ 데이터셋 로드 성공! ---")

# Recover the environment
env = dataset.recover_environment()

# 1. Get all episodes by treating the dataset object as a list
episodes = list(dataset)

# 2. Concatenate the data from each episode
observations = np.concatenate([e.observations for e in episodes])
actions = np.concatenate([e.actions for e in episodes])
rewards = np.concatenate([e.rewards for e in episodes])
terminations = np.concatenate([e.terminations for e in episodes])
truncations = np.concatenate([e.truncations for e in episodes])

# 3. Manually create `next_observations` by shifting the `observations` array
# np.roll with shift=-1 moves each element one position to the front
next_observations = np.roll(observations, -1, axis=0)

# The 'done' flag is the combination of terminations and truncations
dones = terminations | truncations

# Note: For the last step of each episode (where dones[i] is True), 
# the corresponding next_observations[i] is the start of the *next* episode.
# Most RL algorithms handle this correctly by checking the 'done' flag.

print(f"\n데이터셋 ID: {dataset.spec.dataset_id}")
print(f"총 샘플 수: {len(observations)}")

# --- Print first 5 samples ---
print("\n--- 데이터 샘플 (처음 5개) ---")
for i in range(5):
    print(f"[{i+1}]")
    print(f"  - Observation (상태): {observations[i].round(2)}")
    print(f"  - Action (행동): {actions[i].round(2)}")
    print(f"  - Reward (보상): {rewards[i]}")
    # We now use the combined 'dones' flag
    print(f"  - Done (종료 여부): {dones[i]}")
    print(f"  - Next Observation (다음 상태): {next_observations[i].round(2)}")
    print("-" * 20)

env.close()




import minari
import numpy as np

def calculate_average_return(dataset_name: str) -> float:
    """
    데이터셋 이름이 주어지면, 해당 데이터셋의 
    에피소드당 평균 누적 보상(Average Return)을 계산합니다.
    """
    print(f"'{dataset_name}' 데이터셋을 불러오는 중...")
    try:
        # 1. 데이터셋 로드 (로컬에 없으면 다운로드)
        dataset = minari.load_dataset(dataset_name, download=True)
        
        # 2. 모든 에피소드 가져오기
        episodes = list(dataset)
        
        # 3. 각 에피소드의 누적 보상 계산
        #    - episode.rewards는 해당 에피소드의 모든 보상값을 담고 있는 배열입니다.
        #    - np.sum()으로 각 에피소드의 보상 총합을 구합니다.
        episode_returns = [np.sum(episode.rewards) for episode in episodes]
        
        # 4. 모든 에피소드의 누적 보상에 대한 평균 계산
        average_return = np.mean(episode_returns)
        
        print(f"'{dataset_name}' 처리 완료.")
        return average_return

    except Exception as e:
        print(f"'{dataset_name}' 처리 중 오류 발생: {e}")
        return 0.0

# --- 비교 실행 ---
print("데이터셋별 평균 누적 보상 비교를 시작합니다.\n")

# 비교할 데이터셋 이름
medium_dataset_name = "mujoco/hopper/medium-v0"
expert_dataset_name = "mujoco/hopper/expert-v0"

# 각 데이터셋의 평균 누적 보상 계산
avg_return_medium = calculate_average_return(medium_dataset_name)
print("-" * 30)
avg_return_expert = calculate_average_return(expert_dataset_name)

# --- 최종 결과 출력 ---
print("\n--- 최종 비교 결과 ---")
print(f"🏃 Medium 데이터셋의 평균 누적 보상: {avg_return_medium:.2f}")
print(f"🏆 Expert 데이터셋의 평균 누적 보상: {avg_return_expert:.2f}")

if avg_return_expert > avg_return_medium:
    print("\n결론: Expert 정책이 Medium 정책보다 에피소드 전체에서 훨씬 높은 총점을 기록했습니다.")
else:
    print("\n결론: 예상과 달리 Medium 정책의 평균 총점이 더 높게 나왔습니다. (데이터 확인 필요)")





import gymnasium as gym

# 1. Minari 데이터셋 로드
dataset = minari.load_dataset("mujoco/hopper/expert-v0", download=True)
episodes_data = list(dataset)

env = gym.make(dataset.spec.env_spec.id, render_mode='human')  # MuJoCo viewer 실행을 위해 'human' 설정

obs, info = env.reset()
for action in episodes_data[0].actions:
    # action = env.action_space.sample()  # 무작위 행동
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        # obs, info = env.reset()
        break
env.close()

