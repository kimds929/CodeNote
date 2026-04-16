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
# logging.langsmith("Default project", set_enable=False)  # LangSmith 추적을 하지 않습니다.
##############################################################################################################

# llm 객체생성
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser



##############################################################################################################
# ### 방법 1. from_template() 메소드를 사용하여 PromptTemplate 객체 생성
#     - 치환될 변수를 `{ 변수 }` 로 묶어서 템플릿을 정의합니다.


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="gpt-4.1-nano")

from langchain_core.prompts import PromptTemplate

# template 정의. {country}는 변수로, 이후에 값이 들어갈 자리를 의미
template = "{country}의 수도는 어디인가요?"

# from_template 메소드를 이용하여 PromptTemplate 객체 생성
prompt = PromptTemplate.from_template(template)
prompt

# prompt 생성. format 메소드를 이용하여 변수에 값을 넣어줌
prompt = prompt.format(country="대한민국")
prompt



# template 정의
template = "{country}의 수도는 어디인가요?"

# from_template 메소드를 이용하여 PromptTemplate 객체 생성
prompt = PromptTemplate.from_template(template)

# chain 생성
chain = prompt | llm


# country 변수에 입력된 값이 자동으로 치환되어 수행됨
chain.invoke("대한민국").content




# ------------------------------------------------------------------------------------------------
### 방법 2. PromptTemplate 객체 생성과 동시에 prompt 생성
# template 정의
template = "{country}의 수도는 어디인가요?"

# PromptTemplate 객체를 활용하여 prompt_template 생성
prompt = PromptTemplate(
    template=template,
    input_variables=["country"],
)

prompt

# prompt 생성
prompt.format(country="대한민국")


# template 정의
template = "{country1}과 {country2}의 수도는 각각 어디인가요?"

# PromptTemplate 객체를 활용하여 prompt_template 생성
prompt = PromptTemplate(
    template=template,
    input_variables=["country1"],
    partial_variables={
        "country2": "미국"  # dictionary 형태로 partial_variables를 전달
    },
)

prompt.format(country1="대한민국")

prompt_partial = prompt.partial(country2="캐나다")      # 중간과정에서 변수가 변경될 때,
prompt_partial

prompt_partial.format(country1="대한민국")

chain = prompt_partial | llm

chain.invoke("대한민국").content



# ------------------------------------------------------------------------------------------------
# ### `partial_variables`: 부분 변수 채움
#   `partial`을 사용하는 일반적인 용도는 함수를 부분적으로 사용하는 것입니다. 이 사용 사례는 **항상 공통된 방식으로 가져오고 싶은 변수** 가 있는 경우입니다.
#   대표적인 예가 **날짜나 시간** 입니다.
#   항상 현재 날짜가 표시되기를 원하는 프롬프트가 있다고 가정해 보겠습니다. 프롬프트에 하드 코딩할 수도 없고, 다른 입력 변수와 함께 전달하는 것도 번거롭습니다. 이 경우 항상 현재 **날짜를 반환하는 함수** 를 사용하여 프롬프트를 부분적으로 변경할 수 있으면 매우 편리합니다.

from datetime import datetime

# 오늘 날짜를 출력
datetime.now().strftime("%B %d")


# 날짜를 반환하는 함수 정의
def get_today():
    return datetime.now().strftime("%B %d")

prompt = PromptTemplate(
    template="오늘의 날짜는 {today} 입니다. 오늘이 생일인 유명인 {n}명을 나열해 주세요. 생년월일을 표기해주세요.",
    input_variables=["n"],
    partial_variables={
        "today": get_today  # dictionary 형태로 partial_variables를 전달
    },
)

# prompt 생성
prompt.format(n=3)


# chain 을 생성합니다.
chain = prompt | llm

# chain 을 실행 후 결과를 확인합니다.
print(chain.invoke(2).content)


# chain 을 실행 후 결과를 확인합니다.
print(chain.invoke({"today": "Jan 02", "n": 3}).content)




########################################################################################################################
## 파일로부터 template 읽어오기
from langchain_core.prompts import load_prompt

prompt = load_prompt("D:/DataScience/★GitHub_kimds929/CodeNote/56_AgentAI/Prompts_yaml/fruit_color.yaml", encoding="utf-8")
prompt

prompt.format(fruit="사과")

prompt2 = load_prompt("D:/DataScience/★GitHub_kimds929/CodeNote/56_AgentAI/Prompts_yaml/capital.yaml", encoding="utf-8")
print(prompt2.format(country="대한민국"))

chain = prompt2 | llm

print(chain.invoke("대한민국").content)




########################################################################################################################
# ## ChatPromptTemplate
#     `ChatPromptTemplate` 은 대화목록을 프롬프트로 주입하고자 할 때 활용할 수 있습니다.
#        - PromptTemplate → “하나의 문자열 프롬프트”
#           . 초기 LLM (GPT-3 등) → 텍스트 완성 모델
#           . 입력 = 하나의 긴 문자열
#               👉 그래서 단순 템플릿이면 충분 (간단한 작업, 번역, 요약, 한줄질문)
#
#        - ChatPromptTemplate → “대화 구조(역할 포함)를 가진 프롬프트”
#           . Chat 모델 (GPT-4, GPT-4o 등)
#           . 입력 = 메시지 배열
#              👉 역할이 중요해짐: (캐릭터설정, 대화맥락유지, system prompt가 중요할때)
#                  . system → 성격 설정  (모델이 “어떤 방식으로 답해야 하는지”를 정하는 상위 지침, 모델의 “행동 규칙 / 성격 / 정책” 설정, 가장 우선순위가 높음, 전체 응답 스타일에 지속적으로 영향
#                       system이 없으면 → 그냥 일반적인 답
#                       system이 있으면 → 친절하게, 짧게, 전문가처럼, JSON으로, 한국어로, 안전하게 등
#                        → 응답의 성격 자체가 달라진다
#                  . human or user → 질문  (실제 사용자 입력, 질문 / 요청, 모델이 답해야 하는 대상)
#                  . ai or assistant → 이전 답변 (모델의 이전 응답, 대화 맥락 유지, 모델이 참고할 수 있는 정보 제공)
#                  . tool or function → 도구 사용 (agent, tool use에서 중요; 모델이 외부 도구나 함수 호출을 시뮬레이션할 때 사용, 모델이 특정 작업을 수행하도록 지시할 때 활용)
#
#           ※ ChatPromptTemplate 자체는 아무것도 “기억”하지 않는다
#               “이전 대화를 내가 직접 넣어주면 → 모델이 그걸 보고 이어서 답하는 것처럼 행동한다”
            
#     메시지는 튜플(tuple) 형식으로 구성하며, (`role`, `message`) 로 구성하여 리스트로 생성할 수 있습니다.
#     **role**
#     - `"system"`: 시스템 설정 메시지 입니다. 주로 전역설정과 관련된 프롬프트입니다.
#     - `"human"` : 사용자 입력 메시지 입니다.
#     - `"ai"`: AI 의 답변 메시지입니다.

# from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_template("{country}의 수도는 어디인가요?")
chat_prompt

chat_prompt.format(country="대한민국")      # 'Human: 대한민국의 수도는 어디인가요?'



from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        # role, message
        ("system", "당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 {name} 입니다."),
        ("human", "반가워요!"),
        ("ai", "안녕하세요! 무엇을 도와드릴까요?"),
        ("human", "{user_input}"),
    ]
)

# 챗 message 를 생성합니다.
messages = chat_template.format_messages(
    name="Van", user_input="당신의 이름은 무엇입니까?"
)
print(messages)

llm.invoke(messages).content


# Chain
chain = chat_template | llm

chain.invoke({"name": "Van", "user_input": "당신의 이름은 무엇입니까?"}).content



########################################################################################################################
## MessagePlaceholder
#   또한 LangChain은 포맷하는 동안 렌더링할 메시지를 완전히 제어할 수 있는 `MessagePlaceholder` 를 제공합니다. 
#   메시지 프롬프트 템플릿에 어떤 역할을 사용해야 할지 확실하지 않거나 서식 지정 중에 메시지 목록을 삽입하려는 경우 유용할 수 있습니다.


from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 요약 전문 AI 어시스턴트입니다. 당신의 임무는 주요 키워드로 대화를 요약하는 것입니다.",
        ),
        MessagesPlaceholder(variable_name="conversation"),      # 아직 확정된 message는 아니지만, 나중에 채워질 message를 placeholder로 잡아두는 것. (위치만 미리 잡아둠)
        ("human", "지금까지의 대화를 {word_count} 단어로 요약합니다."),
    ]
)
chat_prompt


formatted_chat_prompt = chat_prompt.format(
    word_count=5,
    conversation=[
        ("human", "안녕하세요! 저는 오늘 새로 입사한 Van 입니다. 만나서 반갑습니다."),
        ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
    ],
)
print(formatted_chat_prompt)

formatted_msg_chat_prompt = chat_prompt.format_messages(
    word_count=5,
    conversation=[
        ("human", "안녕하세요! 저는 오늘 새로 입사한 Van 입니다. 만나서 반갑습니다."),
        ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
    ],
)
formatted_msg_chat_prompt


# chain 생성
chain = chat_prompt | llm | StrOutputParser()

# chain 실행 및 결과확인
chain.invoke(
    {
        "word_count": 5,
        "conversation": [
            (
                "human",
                "안녕하세요! 저는 오늘 새로 입사한 Van 입니다. 만나서 반갑습니다.",
            ),
            ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
        ],
    }
)

