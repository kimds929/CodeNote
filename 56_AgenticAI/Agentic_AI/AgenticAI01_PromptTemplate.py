# C:\Users\Admin\AppData\Local\pypoetry\Cache\virtualenvs\langchain-kr-sSe9WGAd-py3.11\Scripts\python.exe
import sys
import os
if os.path.isdir("D:/DataScience/★GitHub_kimds929"):
    library_path = "D:/DataScience/★GitHub_kimds929/DS_Library"
    folder_path = "D:/DataScience/★GitHub_kimds929/CodeNote/56_AgenticAI"
    env_path = 'D:/DataScience/DataBase/Keys/.env'
    llm_case = "ChatOpenAI"
else:
    if os.path.isdir("D:/DataScience/PythonforWork"):
        base_path = "D:/DataScience/PythonforWork"
    elif os.path.isdir("C:/Users/kimds929/DataScience"):
        base_path = "C:/Users/kimds929/DataScience"
    
    library_path = f"{base_path}/DS_Library"
    folder_path = f"{base_path}/AgenticAI"
    env_path = f"{base_path}/AgenticAI/.env"
    llm_case = "PgptLLM"
sys.path.append(library_path)


import requests
import base64
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from DS_AgenticAI import langsmith, PgptLLM, StreamResponse, read_messages, PgptEmbeddings
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

result = load_dotenv(env_path)
print("로드 결과:", result)

##########################################################################################

# LLM 객체 생성
if llm_case == "PgptLLM":
    llm = PgptLLM(
        api_key=os.getenv("API_KEY"),
        emp_no=os.getenv("EMP_NO"),
        # model_name="gpt-4.1-nano",
        model_name="gpt-5-nano",
        # temperature=2.0,  # 정상 코드에 있던 설정값 적용
        # top_p=0.9,
        # stream_usage=True
    )
    embeddings = PgptEmbeddings(
        api_key=os.getenv("API_KEY"),
        emp_no=os.getenv("EMP_NO"),
        model_name='text-embedding-ada-002'
    )
elif llm_case == "ChatOpenAI":
    llm = ChatOpenAI(
        # temperature=0.1,  # 창의성 (0.0 ~ 2.0)
        # model_name="gpt-4o-mini",  # 모델명
        # model_name="gpt-4.1-nano",  # 모델명
        model_name="gpt-5-nano",  # 모델명
    )
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    langsmith("Default project")      # LangSmith 추적을 시작합니다.
    # langsmith("Default project", set_enable=False)  # LangSmith 추적을 하지 않습니다.
print(llm)
print(embeddings)

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################











##########################################################################################
##########################################################################################
# Basic
##########################################################################################
##########################################################################################

from langchain_core.messages import SystemMessage, HumanMessage

messages = [
    SystemMessage(content="당신은 AI 도우미입니다."),
    HumanMessage(content="대한민국 수도는?")
]

# 실행

llm.invoke(messages)
res = StreamResponse(llm.stream(messages))
res.content

# ------------------------------------------------------------------------------------------------

#  LCEL (LangChain Expression Language) 체인 구성 프롬프트와 출력 파서를 연결
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 기획 보고서 작성 전문가입니다."),
    ("user", "{topic}에 대한 보고서 목차를 3가지로 짜주세요.")
])

# 체인 연결 (Prompt -> PoscoGPT -> String Parser)
chain = prompt | llm | StrOutputParser()

# 실행 (invoke, stream, batch 모두 가능)
result = chain.invoke({"topic": "대한민국 2026년 경제 상황 서술"})
print(result)

# ------------------------------------------------------------------------------------------------



from langchain_core.prompts import PromptTemplate

# template 정의. {country}는 변수로, 이후에 값이 들어갈 자리를 의미
template = "{country}의 수도는 어디인가요?"

# from_template 메소드를 이용하여 PromptTemplate 객체 생성
prompt = PromptTemplate.from_template(template)
prompt

# chain 생성
chain = prompt | llm

# country 변수에 입력된 값이 자동으로 치환되어 수행됨
chain.invoke("대한민국").content





##############################################################################################################
# ① System (시스템 역할)
#   . LangChain 클래스: SystemMessagePromptTemplate (또는 튜플에서 "system")
#   . 역할: AI의 페르소나(성격), 행동 지침, 제약 사항, 출력 형식을 전역적으로 설정합니다.
#   . 실무 활용: "당신은 20년 차 시니어 파이썬 개발자입니다", "반드시 JSON 형태로만 응답하세요", "모르는 것은 모른다고 대답하세요"와 같은 핵심 규칙을 부여할 때 사용합니다. 모델은 User의 지시보다 System의 지시를 더 무겁게(우선적으로) 받아들이는 경향이 있습니다.
#
# ② User / Human (사용자 역할)
#   . LangChain 클래스: HumanMessagePromptTemplate (또는 튜플에서 "user", "human")
#   . 역할: 사용자가 AI에게 던지는 실제 질문, 요청, 또는 분석할 데이터를 담습니다.
#   . 실무 활용: "이 코드를 리뷰해줘", "다음 기사를 3줄로 요약해줘"와 같이 매번 바뀌는 동적인 입력값이 주로 이곳에 들어갑니다.
# 
# ③ Assistant / AI (AI 어시스턴트 역할)
#   . LangChain 클래스: AIMessagePromptTemplate (또는 튜플에서 "assistant", "ai")
#   . 역할: AI가 과거에 대답했던 내용을 나타냅니다.
#   . 실무 활용:
#       . 대화 기록(Memory): 챗봇을 만들 때 이전 대화 문맥을 모델에게 알려주기 위해 사용합니다.
#       . Few-shot 예시 제공: AI에게 "이런 식으로 대답해"라는 모범 답안 예시를 미리 보여줄 때 System과 User 메시지 사이에 끼워 넣습니다.
#
# ④ MessagesPlaceholder (특수 역할 - 메시지 묶음)
#   . LangChain 클래스: MessagesPlaceholder
#   . 역할: 길이가 정해지지 않은 여러 개의 메시지 배열(주로 과거 대화 기록)을 통째로 끼워 넣을 빈칸(Placeholder) 역할을 합니다.
#   . 실무 활용: 사용자와 AI가 나눈 수십 번의 대화 기록(Chat History)을 프롬프트 중간에 동적으로 삽입할 때 필수적으로 사용됩니다.



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

chain = chat_template | llm

StreamResponse(chain.stream({'name':'Siri', 'user_input':'당신의 이름은 무엇입니까?'}))







##############################################################################################################
##############################################################################################################
# Prompt Template
##############################################################################################################
##############################################################################################################
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda


# Chain에서의 Runnable
chain1 = (
    {"country": RunnablePassthrough()}
    | PromptTemplate.from_template("{country} 의 수도는?")
    | llm
)
chain2 = (
    {"country": RunnablePassthrough()}
    | PromptTemplate.from_template("{country} 의 면적은?")
    | llm
)

combined_chain = RunnableParallel(capital=chain1, area=chain2)
combined_chain.invoke("대한민국")























