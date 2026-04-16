# C:\Users\Admin\AppData\Local\pypoetry\Cache\virtualenvs\langchain-kr-sSe9WGAd-py3.11\Scripts\python.exe

from dotenv import load_dotenv
import numpy as np
import os

dotenv_path = r'D:\DataScience\DataBase\Keys\.env'
load_dotenv(dotenv_path)
result = load_dotenv(dotenv_path)
print("로드 결과:", result)

print(f"[API KEY]\n{os.environ['OPENAI_API_KEY'][:-15]}" + "*" * 15)

# ---------------------------------------------------------------------------------------------------------

from langchain_teddynote import logging

# 프로젝트 이름을 입력합니다.
logging.langsmith("Default project")
logging.langsmith("Default project", set_enable=False)  # LangSmith 추적을 하지 않습니다.
##############################################################################################################

# llm 객체생성
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

##############################################################################################################

## async stream: 비동기 스트림
# 함수 `chain.astream`은 비동기 스트림을 생성하며, 주어진 토픽에 대한 메시지를 비동기적으로 처리합니다.
# 비동기 for 루프(`async for`)를 사용하여 스트림에서 메시지를 순차적으로 받아오고, `print` 함수를 통해 메시지의 내용(`s.content`)을 즉시 출력합니다. `end=""`는 출력 후 줄바꿈을 하지 않도록 설정하며, `flush=True`는 출력 버퍼를 강제로 비워 즉시 출력되도록 합니다.



# ChatOpenAI 모델을 인스턴스화합니다.
model = ChatOpenAI(model_name="gpt-4.1-nano")

# 주어진 토픽에 대한 농담을 요청하는 프롬프트 템플릿을 생성합니다.
prompt = PromptTemplate.from_template("{topic} 에 대하여 1문장으로 설명해줘.")
# 프롬프트와 모델을 연결하여 대화 체인을 생성합니다.
chain = prompt | model | StrOutputParser()


## async invoke: 비동기 호출
# 비동기 체인 객체의 'ainvoke' 메서드를 호출하여 'NVDA' 토픽을 처리합니다.
my_process = chain.ainvoke({"topic": "NVDA"})       # 답변받을 준비과정

answer = await my_process        # 비동기처리 시작

## async stream: 비동기 스트림
async for token in chain.astream({"topic": "YouTube"}):
    # 메시지 내용을 출력합니다. 줄바꿈 없이 바로 출력하고 버퍼를 비웁니다.
    print(token, end="", flush=True)


## async batch: 비동기 배치
my_abatch_process = chain.abatch(
    [{"topic": "YouTube"}, {"topic": "Instagram"}, {"topic": "Facebook"}]
)
answers = await my_abatch_process




