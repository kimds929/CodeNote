import os
import random
import shutil

# 원본 데이터셋 경로
train_path = r"D:\DataBase\Data_Image\fruit-360-middle\TrainSet"   # 예: dataset/apple, dataset/orange ...
test_path = r"D:\DataBase\Data_Image\fruit-360-middle\TestSet"   # 예: dataset/apple, dataset/orange ...



# (TrainSet to TestSet) ####################################################
# 테스트셋 개수
n_move_to_test = 10

# 라벨(폴더명) 자동 탐색
train_labels = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]

for label in train_labels:
    label_path = os.path.join(train_path, label)
    images = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]
    
    if len(images):
        sample_imgs = random.sample(images, min(n_move_to_test, len(images)))
        
        # TestSet 폴더 경로
        test_label_path = f"{test_path}/{label}"
        os.makedirs(test_label_path, exist_ok=True)

        for img in sample_imgs:
            src = os.path.join(label_path, img)
            dst = os.path.join(test_label_path, img)
            shutil.move(src, dst)
        
        images_aft = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]

        print(f"{label}: {len(sample_imgs)}개 이동 완료 → {test_label_path}")
        print(f"Training_Image : {len(images)} → {len(images_aft)}")



# (TestSet to TrainSet) ####################################################
# 테스트셋 개수
n_move_to_train = 30

# 라벨(폴더명) 자동 탐색
test_labels = [d for d in os.listdir(test_path) if os.path.isdir(os.path.join(train_path, d))]

for label in test_labels:
    label_path = os.path.join(test_path, label)
    images = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]
    
    if len(images):
        # 랜덤 샘플링
        sample_imgs = random.sample(images, min(n_move_to_train, len(images)))
        
        # TraintSet 폴더 경로
        train_label_path = f"{train_path}/{label}"
        os.makedirs(train_label_path, exist_ok=True)
        
        for img in sample_imgs:
            src = os.path.join(label_path, img)
            dst = os.path.join(train_label_path, img)
            shutil.move(src, dst)
        
        images_aft = [f for f in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, f))]

        print(f"{label}: {len(sample_imgs)}개 이동 완료 → {train_label_path}")
        print(f"Test_Image : {len(images)} → {len(images_aft)}")