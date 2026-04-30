import os
current_file_path = os.path.abspath(__file__).replace('\\','/')

if os.path.isdir("D:/DataScience/★GitHub_kimds929"):
    start_script_folder ="D:/DataScience/★GitHub_kimds929/CodeNote/56_AgenticAI"
elif os.path.isdir("D:/DataScience/PythonforWork"):
    start_script_folder ="D:/DataScience/PythonForwork/AgenticAI"
elif os.path.isdir("C:/Users/kimds929/DataScience"):
    start_script_folder = "C:/Users/kimds929/DataScience/AgenticAI"
        
start_script_path = f'{start_script_folder}/StartingScript_AgenticAI.txt'

with open(start_script_path, 'r', encoding='utf-8') as f:
    script = f.read()
script_formatted = script.replace('{base_folder_name}', 'DataScience') \
                         .replace('{current_path}', current_file_path)
# print(script_formatted)
exec(script_formatted)

################################################################################################





##########################################################################################
##########################################################################################
## Hub로부터 Prompt 받아오기
#   https://smith.langchain.com/hub
##########################################################################################
##########################################################################################

from langchain import hub

# 가장 최신 버전의 프롬프트를 가져옵니다.
prompt = hub.pull("rlm/rag-prompt")


# 프롬프트 내용 출력
for prompt_msg in prompt.messages:
    print(prompt_msg.prompt.template)

# 특정 버전의 프롬프트를 가져오려면 버전 해시를 지정하세요
prompt = hub.pull("rlm/rag-prompt:50442af1")
prompt





##########################################################################################
## Prompt Hub 에 자신의 프롬프트 등록
##########################################################################################

from langchain.prompts import ChatPromptTemplate


prompt = ChatPromptTemplate.from_template(
    "주어진 내용을 바탕으로 다음 문장을 요약하세요. 답변은 반드시 한글로 작성하세요\n\nCONTEXT: {context}\n\nSUMMARY:"
)
prompt

# from langchain import hub

# # 프롬프트를 허브에 업로드합니다.
# hub.push("[ID]/[repository-name]", prompt)
