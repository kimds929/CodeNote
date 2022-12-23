# https://coffeedjimmy.github.io/tensorflow/2019/09/14/tf2_using_gpu/

import tensorflow as tf

tf.__version__
tf_gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.debugging.set_log_device_placement(True)     # 해당 연산이 어떤 장치에 할당 되었는지 알려줌
# tf.debugging.set_log_device_placement(False)     # 해당 연산이 어떤 장치에 할당 되었는지 알려줌


# 【 Terminal GPU정보 보기 】
# nvidia-smi
# nvidia-smi -lms 1000      #(1000ms마다 갱신)
# nvcc -V                   # CUDA버전 확인

# 【 GPU 갯수조절 】 ***
if tf_gpu:
    try:
        # 첫 번째 GPU만 사용하도록 제한
        tf.config.experimental.set_visible_devices(tf_gpu[0], 'GPU')
    except RuntimeError as e:
        print(e)

# 【 GPU 메모리 제한 1 】 *** 초기에는 매우 작은 메모리만 할당 해놓고, 필요할 때마다 조금씩 메모리 할당량을 늘려가는 방식
if tf_gpu:
    try:
        tf.config.experimental.set_memory_growth(tf_gpu[0], True)
    except RuntimeError as e:
        print(e)
        
# 【 GPU 메모리 제한2 】 *** 
if tf_gpu:
    # 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
    try:
        tf.config.experimental.set_virtual_device_configuration(
            tf_gpu[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
    except RuntimeError as e:
        print(e)


# 【 Multi-GPU 환경 】 ***
