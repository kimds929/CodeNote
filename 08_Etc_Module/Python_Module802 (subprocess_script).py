import os
base_path = os.getcwd()

import json
import sys
import select
    

# (argument로 바로 전달하기) --------------------------------------------------------------------------------
# if len(sys.argv) > 1:     # 인자가 있는지 확인
#     data = json.loads(sys.argv[1])
#     # print("전달된 인자:", sys.argv[1:])
#     # data['result'] = 'complete'
#     print(data)
# else:
#     raise Exception("Error. Get No Argument")
#     print("error. no argument")


# (argument Input으로 전달하기) ----------------------------------------------------------------------------
# if select.select([sys.stdin], [], [], 0.0)[0]:        # sys.stdin에 데이터가 있는지 확인
#     data = sys.stdin.read()  # 데이터가 있으면 읽기
#     data = json.loads(sys.stdin.read())  # 데이터가 있으면 읽기
#     print(data)
# else:
#     print("error. no argument")

    