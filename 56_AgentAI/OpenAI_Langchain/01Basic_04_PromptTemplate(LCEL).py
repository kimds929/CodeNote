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

llm = ChatOpenAI(
    # temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    # model_name="gpt-4o-mini",  # 모델명
    model_name="gpt-4.1-nano",  # 모델명
    # model_name="gpt-5-nano",  # 모델명
)

##############################################################################################################
### 프롬프트 템플릿의 활용

# `PromptTemplate`
#    - 사용자의 입력 변수를 사용하여 완전한 프롬프트 문자열을 만드는 데 사용되는 템플릿입니다
#    - 사용법
#      - `template`: 템플릿 문자열입니다. 이 문자열 내에서 중괄호 `{}`는 변수를 나타냅니다.
#      - `input_variables`: 중괄호 안에 들어갈 변수의 이름을 리스트로 정의합니다.

# `input_variables`
#   - input_variables는 PromptTemplate에서 사용되는 변수의 이름을 정의하는 리스트입니다.


from langchain_core.prompts import PromptTemplate

# `from_template()` 메소드를 사용하여 PromptTemplate 객체 생성
# template 정의
template = "{country}의 수도는 어디인가요?"

# from_template 메소드를 이용하여 PromptTemplate 객체 생성
prompt_template = PromptTemplate.from_template(template)
prompt_template

# prompt 생성
prompt = prompt_template.format(country="대한민국")
prompt

# prompt 생성
prompt = prompt_template.format(country="미국")
prompt



################################################################################################################
### LCEL(LangChain Expression Language)로 Chain 생성 ############################################################
# ![lcel.png](./images/lcel.png)

# 여기서 우리는 LCEL을 사용하여 다양한 구성 요소를 단일 체인으로 결합합니다
# ★ chain = prompt | model | output_parser
# `|` 기호는 [unix 파이프 연산자](<https://en.wikipedia.org/wiki/Pipeline_(Unix)>)와 유사하며, 서로 다른 구성 요소를 연결하고 한 구성 요소의 출력을 다음 구성 요소의 입력으로 전달합니다.
# 이 체인에서 사용자 입력은 프롬프트 템플릿으로 전달되고, 그런 다음 프롬프트 템플릿 출력은 모델로 전달됩니다. 각 구성 요소를 개별적으로 살펴보면 무슨 일이 일어나고 있는지 이해할 수 있습니다.

# prompt 를 PromptTemplate 객체로 생성합니다.
prompt = PromptTemplate.from_template("최소한의 답변만 짧게해줘. {topic}에 대해 {how}쉽게 설명해주세요.")
model = ChatOpenAI(model="gpt-4.1-nano", temperature=0.1)
chain = prompt | model


# invoke() 호출
# input 딕셔너리에 주제를 '인공지능 모델의 학습 원리'으로 설정합니다.
input = {"topic": "인공지능 모델의 학습 원리", "how": "쉽게"}     # 반드시 template에서 정의한 변수명과 일치해야 합니다. (해당 변수명이 정의되지 않으면 오류가 발생)

# prompt 객체와 model 객체를 파이프(|) 연산자로 연결하고 invoke 메서드를 사용하여 input을 전달합니다.
# 이를 통해 AI 모델이 생성한 메시지를 반환합니다.
response = chain.invoke(input)
print(response.content)

# 스트리밍 출력을 위한 요청
answer = chain.stream(input)
# 스트리밍 출력
from langchain_teddynote.messages import stream_response

response_output = stream_response(answer, return_output=True)


# ----------------------------------------------------------------------------------------------------
# 출력 Parser (Output Parser) : 답변형식을 변경
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()   # 문자열 형식으로 바꿔줌

prompt = PromptTemplate.from_template("최소한의 답변만 짧게해줘. {topic}에 대해 설명해주세요.")

# 프롬프트, 모델, 출력 파서를 연결하여 처리 체인을 구성합니다.
chain = prompt | model | output_parser


input = {"topic": "인공지능 모델의 학습 원리"}

# chain 객체의 invoke 메서드를 사용하여 input을 전달합니다.
chain.invoke(input)


# 스트리밍 출력을 위한 요청
answer = chain.stream(input)
# 스트리밍 출력
response_output = stream_response(answer, return_output=True)


################################################################################################
### 템플릿을 변경하여 적용
# - 아래의 프롬프트 내용을 얼마든지 **변경** 하여 테스트 해볼 수 있습니다.
# - `model_name` 역시 변경하여 테스트가 가능합니다.


template = """
당신은 영어를 가르치는 10년차 영어 선생님입니다. 주어진 상황에 맞는 영어 회화를 작성해 주세요.
양식은 [FORMAT]을 참고하여 작성해 주세요.

#상황:
{question}

#FORMAT:
- 영어 회화:
- 한글 해석:
"""

# 프롬프트 템플릿을 이용하여 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(template)

# ChatOpenAI 챗모델을 초기화합니다.
model = ChatOpenAI(model_name="gpt-4.1-nano")

# 문자열 출력 파서를 초기화합니다.
output_parser = StrOutputParser()

# 체인을 구성합니다.
chain = prompt | model | output_parser

# 완성된 Chain을 실행하여 답변을 얻습니다.
print(chain.invoke({"question": "저는 식당에 가서 음식을 주문하고 싶어요"}))


# 완성된 Chain을 실행하여 답변을 얻습니다.
# 스트리밍 출력을 위한 요청
answer = chain.stream({"question": "저는 식당에 가서 음식을 주문하고 싶어요"})
# 스트리밍 출력
response_output = stream_response(answer, return_output=True)


# 이번에는 question 을 '미국에서 피자 주문'으로 설정하여 실행합니다.
# 스트리밍 출력을 위한 요청
answer = chain.stream({"question": "미국에서 피자 주문"})
# 스트리밍 출력
response_output = stream_response(answer, return_output=True)


################################################################################################
## LCEL 인터페이스
# 사용자 정의 체인을 가능한 쉽게 만들 수 있도록, [`Runnable`](https://api.python.langchain.com/en/stable/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable) 프로토콜을 구현했습니다. 

# `Runnable` 프로토콜은 대부분의 컴포넌트에 구현되어 있습니다.

# 이는 표준 인터페이스로, 사용자 정의 체인을 정의하고 표준 방식으로 호출하는 것을 쉽게 만듭니다.
# 표준 인터페이스에는 다음이 포함됩니다.

# - [`stream`](#stream): 응답의 청크를 스트리밍합니다.
# - [`invoke`](#invoke): 입력에 대해 체인을 호출합니다.
# - [`batch`](#batch): 입력 목록에 대해 체인을 호출합니다.

# 비동기 메소드도 있습니다.

# - [`astream`](#async-stream): 비동기적으로 응답의 청크를 스트리밍합니다.
# - [`ainvoke`](#async-invoke): 비동기적으로 입력에 대해 체인을 호출합니다.
# - [`abatch`](#async-batch): 비동기적으로 입력 목록에 대해 체인을 호출합니다.
# - [`astream_log`](#async-stream-intermediate-steps): 최종 응답뿐만 아니라 발생하는 중간 단계를 스트리밍합니다.



# ChatOpenAI 모델을 인스턴스화합니다.
model = ChatOpenAI(model_name="gpt-4.1-nano")

# 주어진 토픽에 대한 농담을 요청하는 프롬프트 템플릿을 생성합니다.
prompt = PromptTemplate.from_template("{topic} 에 대하여 1문장으로 설명해줘.")
# 프롬프트와 모델을 연결하여 대화 체인을 생성합니다.
chain = prompt | model | StrOutputParser()

# (Stream : 실시간 출력)
# chain.stream 메서드를 사용하여 '멀티모달' 토픽에 대한 스트림을 생성하고 반복합니다.
for token in chain.stream({"topic": "멀티모달"}):
    # 스트림에서 받은 데이터의 내용을 출력합니다. 줄바꿈 없이 이어서 출력하고, 버퍼를 즉시 비웁니다.
    print(token, end="", flush=True)
    
# (Invoke : 한번에 출력)
# chain 객체의 invoke 메서드를 호출하고, 'ChatGPT'라는 주제로 딕셔너리를 전달합니다.
answer = chain.invoke({"topic": "ChatGPT"})
print(answer)

# (Batch: 배치(단위 실행) : 동일한 형태의 template을 갖는 여러 입력을 한 번에 처리)
# 주어진 토픽 리스트를 batch 처리하는 함수 호출
answers = chain.batch(
    [
        {"topic": "ChatGPT"},
        {"topic": "Instagram"},
        {"topic": "멀티모달"},
        {"topic": "프로그래밍"},
        {"topic": "머신러닝"},
    ],
    config={"max_concurrency": 3},      # max_concurrency : 동시에 처리가능한 최대 작업 수 
)
answers[0]