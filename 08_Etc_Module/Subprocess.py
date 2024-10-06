import os
base_path = os.getcwd()

import json
import subprocess
import sys

# args = {'init':  True, 'key1': 'value1', 'key2': 'value2'}
args =  {}

# (argument로 바로 전달하기) --------------------------------------------------------------------------------
# result = subprocess.run(['python', f"{base_path}/Subprocess_script.py", json.dumps(args)], capture_output=True, text=True)

# (argument Input으로 전달하기) ----------------------------------------------------------------------------
result = subprocess.run(['python', f"{base_path}/Subprocess_script.py"],
                     input=json.dumps(args), capture_output=True, text=True)
# result = subprocess.run(['python', f"{base_path}/Subprocess_script.py"],
#                      input=json.dumps({}), capture_output=True, text=True)


###################################################################################################
if result.returncode == 0:
    output = result.stdout
    print("script.py에서 반환된 결과:")
    print(output)
else:
    print("script.py 실행 중 에러 발생:")
    print(result.stderr.strip())
# eval(output.split('\n')[0])
###################################################################################################