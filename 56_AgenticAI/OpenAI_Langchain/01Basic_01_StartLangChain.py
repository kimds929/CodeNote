# C:\Users\Admin\AppData\Local\pypoetry\Cache\virtualenvs\langchain-kr-sSe9WGAd-py3.11\Scripts\python.exe

from dotenv import load_dotenv
import numpy as np
import os

dotenv_path = r'D:\DataScience\DataBase\Keys\.env'
load_dotenv(dotenv_path)
result = load_dotenv(dotenv_path)
print("로드 결과:", result)


print(f"[API KEY]\n{os.environ['OPENAI_API_KEY'][:-15]}" + "*" * 15)
##############################################################################################################


# ---------------------------------------------------------------------------------------------------------
from importlib.metadata import version

print("[LangChain 관련 패키지 버전]")
for package_name in [
    "langchain",
    "langchain-core",
    "langchain-experimental",
    "langchain-community",
    "langchain-openai",
    "langchain-teddynote",
    "langchain-huggingface",
    "langchain-google-genai",
    "langchain-anthropic",
    "langchain-cohere",
    "langchain-chroma",
    "langchain-elasticsearch",
    "langchain-upstage",
    "langchain-cohere",
    "langchain-milvus",
    "langchain-text-splitters",
]:
    try:
        package_version = version(package_name)
        print(f"{package_name}: {package_version}")
    except ImportError:
        print(f"{package_name}: 설치되지 않음")
        
# ---------------------------------------------------------------------------------------------------------
    

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# .env 파일에 LANGCHAIN_API_KEY를 입력합니다.
# !pip install -qU langchain-teddynote
# C:\Users\Admin\AppData\Local\pypoetry\Cache\virtualenvs\langchain-kr-sSe9WGAd-py3.11\Lib\site-packages\langchain_teddynote
from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("Default project")
# logging.langsmith("Default project", set_enable=False)  # LangSmith 추적을 하지 않습니다.


##############################################################################################################
### 답변의 형식(AI Message)
from langchain_openai import ChatOpenAI

# 객체 생성
llm = ChatOpenAI(
    # temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    # model_name="gpt-4o-mini",  # 모델명
    model_name="gpt-4.1-nano-nano",  # 모델명
    # model_name="gpt-5-nano",  # 모델명
)

# 질의내용
question = "대한민국의 수도는 어디인가요?"

# ---------------------------------------------------------------------------------------------------------
# imvoke : 출력이 끝날때까지 기다렸다가 한번에 return
response = llm.invoke(question)
# 질의
print(f"[답변]: {response}")
response.content
response.response_metadata
response.response_metadata['token_usage']['total_tokens']       # 입력(prompt) + 출력(response) 모두를 합친 총 토큰 수


### LogProb 활성화
# 주어진 텍스트에 대한 모델의 **토큰 확률의 로그 값** 을 의미합니다. 토큰이란 문장을 구성하는 개별 단어나 문자 등의 요소를 의미하고, 확률은 **모델이 그 토큰을 예측할 확률**을 나타냅니다.

# 객체 생성
llm_with_logprob = ChatOpenAI(
    temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    max_tokens=2048,  # 최대 토큰수
    model_name="gpt-4.1-nano",  # 모델명
).bind(logprobs=True)


# 질의
response = llm_with_logprob.invoke(question)
response.content
response.response_metadata['logprobs']
response.response_metadata['token_usage']['total_tokens']



# ---------------------------------------------------------------------------------------------------------
# streaming : 한 token씩 출력이 생성되는 대로 반환하는 방식
import time
response = llm.stream(question)     # return : generator

response_all = ""
tokens = []
for token in response:
    print(token.content, end="", flush=True)        #  `flush=True` 인자는 출력 버퍼를 즉시 비우도록 한다.
    
    response_all += token.content
    tokens.append(token)
    time.sleep(0.05)  # 토큰이 생성되는 속도를 시뮬레이션하기 위해 잠시 대기


print(f"[스트리밍 답변]: {response_all}")
from langchain_teddynote.messages import stream_response

response = llm.stream(question)
response_all = stream_response(response, return_output=True)
response_all

