# C:\Users\Admin\AppData\Local\pypoetry\Cache\virtualenvs\langchain-kr-sSe9WGAd-py3.11\Scripts\python.exe

import sys
folder_path = "D:/DataScience/★GitHub_kimds929/CodeNote/56_AgenticAI"
sys.path.append("D:/DataScience/★GitHub_kimds929/DS_Library")

import os
import requests
import base64
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

try:
    from DS_AgenticAI import logging, PgptLLM, StreamResponse, read_messages, PgptEmbeddings
except:
    remote_library_url = 'https://raw.githubusercontent.com/kimds929/'
    try:
        import httpimport
        with httpimport.remote_repo(f"{remote_library_url}/DS_Library/main/"):
            from DS_AgenticAI import logging, PgptLLM, StreamResponse, read_messages, PgptEmbeddings
    except:
        import requests
        response = requests.get(f"{remote_library_url}/DS_Library/main/DS_AgenticAI.py", verify=False)
        exec(response.text)


if os.path.exists(f"{folder_path}/.env"):
    result = load_dotenv(dotenv_path=f"{folder_path}/.env")
else:
    result = load_dotenv('D:/DataScience/DataBase/Keys/.env')
print("로드 결과:", result)

##########################################################################################

# LLM 객체 생성
try:
    llm = PgptLLM(
        api_key=os.getenv("API_KEY"),
        emp_no=os.getenv("EMP_NO"),
        model_name="gpt-4.1-nano",
        # temperature=2.0,  # 정상 코드에 있던 설정값 적용
        # top_p=0.9,
        # stream_usage=True
    )
except:
    llm = ChatOpenAI(
        # temperature=0.1,  # 창의성 (0.0 ~ 2.0)
        # model_name="gpt-4o-mini",  # 모델명
        model_name="gpt-4.1-nano",  # 모델명
        # model_name="gpt-5-nano",  # 모델명
    )

    logging.langsmith("Default project")      # LangSmith 추적을 시작합니다.
    # logging.langsmith("Default project", set_enable=False)  # LangSmith 추적을 하지 않습니다.
print(llm)

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################









# llm 객체생성
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser





##############################################################################################################
# Runnable : 체인과 "같은 실행 가능한 객체"를 나타내는 인터페이스입니다.
#           서로 다른 기능들을 레고 블록처럼 쉽게 연결할 수 있게 만들어주는 '표준 접속 규격'"
#   . 특징
#       1. 파이프(|) 기호로 간단한 조립 : `chain = component1 | component2 | component3`
#       2. 통일된 작동 명령어 : `Runnable` 객체는 `invoke`, `stream`, `batch` 등의 메서드를 통해 실행할 수 있습니다.
#   . 다양한 변형도구 제공
#       1. `RunnableSequence` : 여러 `Runnable` 객체를 순차적으로 실행할 수 있게 해주는 도구입니다.
#       2. `RunnableParallel` : 여러 `Runnable` 객체를 병렬로 실행할 수 있게 해주는 도구입니다.
#       3. `RunnablePassthrough` : 입력을 그대로 출력으로 전달하는 도구입니다. 주로 다른 `Runnable` 객체와 함께 사용되어, 입력을 중간 단계로 전달하는 역할을 합니다.

##############################################################################################################
## 데이터를 효과적으로 전달하는 방법
#    - `RunnablePassthrough` 는 입력을 변경하지 않거나 추가 키를 더하여 전달할 수 있습니다. 
#    - `RunnablePassthrough()` 가 단독으로 호출되면, 단순히 입력을 받아 그대로 전달합니다.
#    - `RunnablePassthrough.assign(...)` 방식으로 호출되면, 입력을 받아 assign 함수에 전달된 추가 인수를 추가합니다.



# prompt 와 llm 을 생성합니다.
prompt = PromptTemplate.from_template("{num} 의 10배는?")
llm = ChatOpenAI(model_name="gpt-4.1-nano")

# chain 을 생성합니다.
chain = prompt | llm

# chain 을 실행합니다.
chain.invoke({"num": 5})

chain.invoke(5) 


## RunnablePathrough : 입력을 그대로 출력으로 전달하는 도구입니다. 주로 다른 `Runnable` 객체와 함께 사용되어, 입력을 중간 단계로 전달하는 역할을 합니다.
#   필요성 : LangChain / LCEL 구조는 기본적으로 이렇게 흐른다: 입력 → 변환 → 변환 → 변환 → 출력
#       근데 문제는: 👉 중간에 원본 데이터를 다시 써야 하는 경우가 매우 많다
from langchain_core.runnables import RunnablePassthrough

# runnable
RunnablePassthrough().invoke({"num": 10})

runnable_chain = {"num": RunnablePassthrough()} | prompt | ChatOpenAI()

# dict 값이 RunnablePassthrough() 로 변경되었습니다.
runnable_chain.invoke(10)

# `RunnablePassthrough.assign()`
#   - 입력 값으로 들어온 값의 key/value 쌍과 새롭게 할당된 key/value 쌍을 합칩니다.
# 입력 키: num, 할당(assign) 키: new_num
(RunnablePassthrough.assign(new_num=lambda x: x["num"] * 3)).invoke({"num": 1})







##############################################################################################################
## Parallel: 병렬성
# LangChain Expression Language가 병렬 요청을 지원하는 방법을 살펴봅시다.
# 예를 들어, `RunnableParallel`을 사용할 때, 각 요소를 병렬로 실행합니다.

# `langchain_core.runnables` 모듈의 `RunnableParallel` 클래스를 사용하여 두 가지 작업을 병렬로 실행하는 예시를 보여줍니다.
# `ChatPromptTemplate.from_template` 메서드를 사용하여 주어진 `country`에 대한 **수도** 와 **면적** 을 구하는 두 개의 체인(`chain1`, `chain2`)을 만듭니다.
# 이 체인들은 각각 `llm`과 파이프(`|`) 연산자를 통해 연결됩니다. 마지막으로, `RunnableParallel` 클래스를 사용하여 이 두 체인을 `capital`와 `area`이라는 키로 결합하여 동시에 실행할 수 있는 `combined` 객체를 생성합니다.


from langchain_core.runnables import RunnableParallel


llm = ChatOpenAI(model_name="gpt-4.1-nano")

# {country} 의 수도를 물어보는 체인을 생성합니다.
chain1 = (
    PromptTemplate.from_template("{country} 의 수도는 어디야?")
    | llm
    | StrOutputParser()
)

# {country} 의 면적을 물어보는 체인을 생성합니다.
chain2 = (
    PromptTemplate.from_template("{country} 의 면적은 얼마야?")
    | llm
    | StrOutputParser()
)

# 위의 2개 체인을 동시에 생성하는 병렬 실행 체인을 생성합니다.
combined = RunnableParallel(capital=chain1, area=chain2)


# chain1 를 실행합니다.
chain1.invoke({"country": "대한민국"})      # '대한민국의 수도는 서울입니다.'

# chain2 를 실행합니다.
chain2.invoke({"country": "미국"})      # '미국의 면적은 약 9.8백만 제곱킬로미터입니다.'


# 병렬 실행 체인을 실행합니다.
combined.invoke({"country": "대한민국"})        # {'capital': '대한민국의 수도는 서울입니다.', 'area': '대한민국의 면적은 약 100,210 제곱킬로미터입니다.'}


# -----------------------------------------------------------------------------------------------------------
# 배치에서의 병렬처리
#   병렬 처리는 다른 실행 가능한 코드와 결합될 수 있습니다.
#   `chain1.batch` 함수는 여러 개의 딕셔너리를 포함하는 리스트를 인자로 받아, 각 딕셔너리에 있는 "topic" 키에 해당하는 값을 처리합니다. 이 예시에서는 "대한민국"와 "미국"라는 두 개의 토픽을 배치 처리하고 있습니다.

# 배치 처리를 수행합니다.
chain1.batch([{"country": "대한민국"}, {"country": "미국"}])    #   ['대한민국의 수도는 서울입니다.', '미국의 수도는 워싱턴 D.C.입니다.']

# 배치 처리를 수행합니다.
chain2.batch([{"country": "대한민국"}, {"country": "미국"}])    #   ['대한민국의 면적은 약 100,210 제곱킬로미터입니다.', '미국의 면적은 약 9.8백만 제곱킬로미터입니다.']

# 주어진 데이터를 배치로 처리합니다.
combined.batch([{"country": "대한민국"}, {"country": "미국"}])    
#   [{'capital': '대한민국의 수도는 서울입니다.', 'area': '대한민국의 면적은 약 100,210 제곱킬로미터입니다.'},
#   {'capital': '미국의 수도는 워싱턴 D.C.입니다.', 'area': '미국의 면적은 약 9.8백만 제곱킬로미터입니다.'}]





# -------------------------------------------------------------------------------------------------------------

# prompt 와 llm 을 생성합니다.
prompt = PromptTemplate.from_template("{num} 의 10배는?")
llm = ChatOpenAI(model_name="gpt-4.1-nano")


# RunnableParallel 인스턴스를 생성합니다. 이 인스턴스는 여러 Runnable 인스턴스를 병렬로 실행할 수 있습니다.
runnable = RunnableParallel(
    # RunnablePassthrough 인스턴스를 'passed' 키워드 인자로 전달합니다. 이는 입력된 데이터를 그대로 통과시키는 역할을 합니다.
    passed=RunnablePassthrough(),
    # 'extra' 키워드 인자로 RunnablePassthrough.assign을 사용하여, 'mult' 람다 함수를 할당합니다. 이 함수는 입력된 딕셔너리의 'num' 키에 해당하는 값을 3배로 증가시킵니다.
    extra=RunnablePassthrough.assign(mult=lambda x: x["num"] * 3),
    # 'modified' 키워드 인자로 람다 함수를 전달합니다. 이 함수는 입력된 딕셔너리의 'num' 키에 해당하는 값에 1을 더합니다.
    modified=lambda x: x["num"] + 1,
)
runnable

# runnable 인스턴스에 {'num': 1} 딕셔너리를 입력으로 전달하여 invoke 메소드를 호출합니다.
runnable.invoke({"num": 1})


# Chain에서의 Runnable
chain1 = (
    {"country": RunnablePassthrough()}
    | PromptTemplate.from_template("{country} 의 수도는?")
    | ChatOpenAI()
)
chain2 = (
    {"country": RunnablePassthrough()}
    | PromptTemplate.from_template("{country} 의 면적은?")
    | ChatOpenAI()
)

combined_chain = RunnableParallel(capital=chain1, area=chain2)
combined_chain.invoke("대한민국")




##############################################################################################################
## RunnableLambda
#   RunnableLambda 를 사용하여 사용자 정의 함수를 맵핑할 수 있습니다.
from datetime import datetime


def get_today(_=None):
    print(f"입력받은 값 : {_}")
    # 오늘 날짜를 가져오기
    return datetime.today().strftime("%b-%d")


# 오늘 날짜를 출력
get_today()

from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# prompt 와 llm 을 생성합니다.
prompt = PromptTemplate.from_template(
    "{today} 가 생일인 유명인 {n} 명을 나열하세요. 생년월일을 표기해 주세요."
)
llm = ChatOpenAI(temperature=0, model_name="gpt-4.1-mini")

# chain 을 생성합니다.
chain = (
    {"today": RunnableLambda(get_today), "n": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 출력
print(chain.invoke(3))

# 출력2
print(chain.invoke({'n': 3}))


# `itemgetter` 를 사용하여 특정 키를 추출합니다.
from operator import itemgetter

# chain 을 생성합니다.
chain = (
    {"today": RunnableLambda(get_today), "n": itemgetter("n")}
    | prompt
    | llm
    | StrOutputParser()
)

# 출력2
print(chain.invoke({'n': 3}))




# 문장의 길이를 반환하는 함수입니다.
def length_function(text):
    return len(text)


# 두 문장의 길이를 곱한 값을 반환하는 함수입니다.
def _multiple_length_function(text1, text2):
    return len(text1) * len(text2)


# _multiple_length_function 함수를 사용하여 두 문장의 길이를 곱한 값을 반환하는 함수입니다.
def multiple_length_function(_dict):
    return _multiple_length_function(_dict["text1"], _dict["text2"])


prompt = ChatPromptTemplate.from_template("{a} + {b} 는 무엇인가요?")
llm = ChatOpenAI()

chain1 = prompt | llm

chain = (
    {
        "a": itemgetter("word1") | RunnableLambda(length_function),
        "b": {"text1": itemgetter("word1"), "text2": itemgetter("word2")}
        | RunnableLambda(multiple_length_function),
    }
    | prompt
    | llm
)

chain.invoke({"word1": "hello", "word2": "world"})