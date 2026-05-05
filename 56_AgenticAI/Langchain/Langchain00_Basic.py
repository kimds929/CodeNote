import sys
import os
import requests
import base64
import json
from dotenv import load_dotenv

folder_path = "D:/DataScience/PythonForwork/Pgpt"
load_dotenv(dotenv_path=f"{folder_path}/.env")


auth_data = {
    "apiKey": os.getenv("API_KEY"),
    "empNo": os.getenv("EMP_NO"),
    "compNo": os.getenv("COMP_NO")
}
token = base64.b64encode(json.dumps(auth_data).encode('utf-8')).decode('utf-8')

################################################################################

url = os.getenv("INVOKE_API_URL")
# url = os.getenv("STREAM_API_URL")


# 2. 헤더 및 데이터 설정
# 엔드포인트 :  API 서비스를 이용하기 위해 데이터를 보내고 받는 '서버의 특정 주소(URL)'
    # 일반 API : 서버가 모든 계산과 처리를 끝낸 후, 결과값(Response)을 한 번에 모두 반환하는 방식 
    #           응답 시간이 짧은 작업, 단순 데이터 추출, 짧은 요약 등 결과를 즉시 받아야 하는 경우에 적합
    # 스트리밍 API : 서버에서 답변이 생성되는 즉시 실시간으로 데이터를 조금씩 전달받는 방식입니다. ChatGPT 서비스처럼 글자가 타자 치듯 나타나는 효과를 구현가능
    #           긴 글을 생성하는 경우, 사용자에게 '답변 중'이라는 느낌을 주어 대기 시간을 심리적으로 줄여야 할 때 적합.

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# -----------------------------------------------------------------------------
data = {
    "model": "gpt-4.1-nano",
    # "model": "gpt-5.2-chat",
    # "model": "gpt-5-nano",
    "messages": [
        {"role": "system", "content": "당신은 AI 도우미입니다."},
        {"role": "system", "content": "Test중이라 답변은 최소한으로 요약해서 최대한 짧게 해줘."},
        {"role": "user", "content": "대한민국의 수도는?"}
    ],
    "temperature": 2.0,        # 0.0~2.0, 창의성 조절 (낮을수록 보수적/정확함) 
    # "max_tokens": 100,         # 답변의 최대 길이 제한 (5.4불가)
    "top_p": 0.9,              # 확률 질량의 합을 기준으로 후보군을 정하는 방식(다른 창의성) : top_p=0.9라면, 확률 합계가 상위 90%가 되는 단어들 중에서만 선택
    "stream": False,           # True로 설정 시 실시간 답변 스트리밍 : 변을 생성하는 즉시 실시간으로 응답을 받아올지(True), 완료될 때까지 기다렸다가 한 번에 받을지(False)를 결정
    "need_origin": True        # PGPT API 서버에서 별도로 정의한 사용자 정의 매개변수
}


# 3. 요청 및 결과 출력
# response = requests.post(url, headers=headers, json=data)
response = requests.post(url, headers=headers, json=data, stream=True)
print(response)


# JSON을 예쁘게 출력 (들여쓰기 4칸, 한글 인코딩 유지)
# print(json.dumps(response.json(), indent=4, ensure_ascii=False))
print(response.json()['choices'][0]['message']['content'])






################################################################################
# 【 LLM MODELS 】
# [{'이름': 'GPT-5.2 Series', '채팅 모델': 'gpt-5.2-chat', '추론 모델': 'gpt-5.2'},
#  {'이름': 'GPT-5.1 Series', '채팅 모델': 'gpt-5.1-chat', '추론 모델': 'gpt-5.1-codex'},
#  {'이름': 'GPT-5 Series', '채팅 모델': 'gpt-5-chat', '추론 모델': 'gpt-5'},
#  {'이름': 'GPT-5 Edge (Mini/Nano)',
#   '채팅 모델': '-',
#   '추론 모델': 'gpt-5-mini\ngpt-5-nano'},
#  {'이름': 'GPT-4.1 Series', '채팅 모델': 'gpt-4.1', '추론 모델': '-'},
#  {'이름': 'GPT-4o Series', '채팅 모델': 'gpt-4o', '추론 모델': 'gpt-4o-mini'}]

# 1. 채팅 모델 (Chat Model)
#   우리가 흔히 아는 ChatGPT(GPT-3.5, GPT-4o), Claude 3.5 Sonnet 등이 여기에 속합니다.
#       . 작동 방식: 사용자의 질문을 받으면, 직관적인 패턴 매칭을 통해 가장 자연스러운 다음 단어(Next Token)를 즉시 예측하여 출력합니다.
#       . 주요 특징:
#           . 빠른 속도: 질문하자마자 실시간으로 글자가 타라락 써집니다.
#           . 지시사항 준수: "JSON 형태로 출력해", "3줄로 요약해", "친절한 톤으로 말해" 같은 형식(Format) 지시를 매우 잘 따릅니다.
#           . 문맥 유지: 대화의 흐름(System, User, Assistant 역할)을 기억하고 티키타카를 하는 데 최적화되어 있습니다.
#       . 단점 (한계):
#           .   복잡한 수학 문제나 다단계 논리 퍼즐을 주면, 깊게 생각하지 않고 '그럴싸해 보이는 오답'을 즉시 뱉어내는 환각(Hallucination) 현상이 잦습니다.
#       . 실무 활용처:
#           . RAG (검색 증강 생성): 검색된 사내 문서를 바탕으로 사용자에게 깔끔하게 요약해서 답변할 때.
#           . 고객 CS 챗봇: 빠른 응답 속도가 생명인 서비스.
#           . 번역 및 텍스트 교정.
# 
# 2. 추론 모델 (Reasoning Model)
#   최근에 발표된 OpenAI의 o1, o3-mini, 그리고 DeepSeek-R1 등이 대표적인 추론 모델입니다.

#   . 작동 방식: 답변을 바로 뱉어내지 않습니다. 내부적으로 **'생각의 사슬(Chain of Thought, CoT)'**이라는 과정을 거칩니다. 문제를 여러 단계로 쪼개고, 스스로 가설을 세우고, 틀리면 다시 돌아가서 수정하는 과정을 거친 후 최종 답변만 출력합니다.
#               예를 들어, 복잡한 논리 문제를 풀 때 내부적으로 다음과 같은 검증 과정을 거칩니다.
#                   P → Q, -Q ☞ -P
#               이러한 대우명제와 같은 논리적 단계를 스스로 수십 번 반복하며 정답을 찾아냅니다.
#   . 주요 특징:
#       . 압도적인 문제 해결력: 복잡한 코딩, 고난도 수학, 난해한 데이터 분석 로직을 짜는 데 있어 채팅 모델과 비교할 수 없을 정도로 뛰어납니다.
#       . 프롬프트 엔지니어링 최소화: "단계별로 생각해(Think step by step)"라고 지시할 필요가 없습니다. 알아서 깊게 생각합니다.
#   . 단점 (한계):
#       . 느린 속도와 높은 비용: 생각하는 시간(Thinking time)이 짧게는 10초에서 길게는 몇 분까지 걸립니다. 그만큼 API 호출 비용(토큰)도 많이 듭니다.
#       . 형식 무시: "반드시 JSON으로만 답해"라고 해도, 자기 생각 과정을 주저리주저리 늘어놓느라 형식을 깨뜨리는 경우가 종종 있습니다.
#   . 실무 활용처:
#       . 복잡한 데이터 분석 (Pandas Agent): 앞서 질문하신 데이터프레임 분석 시, 복잡한 상관관계나 재무 수식을 코드로 짜야 할 때.
#       . 소프트웨어 개발: 수백 줄의 코드를 디버깅하거나 아키텍처를 설계할 때.




################################################################################
# token = base64.b64encode(json.dumps(auth_data).encode('utf-8')).decode('utf-8')

# headers = {
#     "Authorization": f"Bearer {token}",
#     "Content-Type": "application/json"
# }


# from langchain_openai import ChatOpenAI
# from langchain_core.messages import SystemMessage, HumanMessage
# llm = ChatOpenAI(
#     model= "gpt-4.1-nano",
#     base_url= 'http://pgpt.posco.com/s0la01-gpt/gptApi/personalApi/chat/completions',
#     api_key='dummy_key',
#     default_headers=headers,
#     model_kwargs={
#         "extra_body": {
#             "need_origin": True
#         }
#     }
    
# )
# messages = [
#     SystemMessage(content="당신은 AI 도우미입니다."),
#     HumanMessage(content="포스코에 대해서 설명해줘")
# ]
# llm.invoke(messages)





################################################################################################
# 기본 출력
import sys
sys.path.append(folder_path)
from DS_AgentAI import PgptLLM, StreamResponse, read_messages, PgptEmbeddings


# ----------------------------------------------------------------------------------------


# 객체 생성 (본인의 직번과 키를 정확히 입력하세요)
llm = PgptLLM(
    api_key=os.getenv("API_KEY"),
    emp_no=os.getenv("EMP_NO"),
    model_name="gpt-4.1-nano",
    # temperature=2.0,  # 정상 코드에 있던 설정값 적용
    # top_p=0.9,
    # stream_usage=True
)

messages = '대한민국의 수도는?'

# Invoke
res = llm.invoke(messages)
res


# Streaming 
import time
response = llm.stream(messages)     # return : generator

response_all = ""
tokens = []
for token in response:

    print(token.content, end="", flush=True)        #  `flush=True` 인자는 출력 버퍼를 즉시 비우도록 한다.
    
    response_all += token.content
    tokens.append(token)
    time.sleep(0.05)  # 토큰이 생성되는 속도를 시뮬레이션하기 위해 잠시 대기
    

# Streaming Class
result = StreamResponse(llm.stream('서울에 대해 알려줘.') )
result.response
print(result.content)
result.metadata


################################################################################################
# 파일 불러와서 답변 요청하기
with open(f"{folder_path}/message.txt", 'r', encoding='utf-8-sig') as f:
    messages = f.read()
res = StreamResponse(llm.stream(messages))

messages = read_messages(f"{folder_path}/message.txt")
res = StreamResponse(llm.stream(messages))





################################################################################################
# LCEL : LangChain Expression Language
#   LangChain 프레임워크에서 AI 모델, 프롬프트, 출력 파서(Parser) 등 다양한 구성 요소를 **레고 블록처럼 쉽고 직관적으로 연결(Chaining)할 수 있게 해주는 선언적 언어(Declarative Language)
#   1. 핵심 목적
#   - 사용자들이 자연어 처리를 위한 작업을 쉽게 정의하고 조합할 수 있도록 지원.
#   - 코드 작성의 복잡성을 줄이고, 선언적 방식으로 체인 또는 작업을 구성.
#
#   2. LCEL의 핵심 개념: 파이프 연산자 (|)
#   - 유닉스(Unix) 파이프라인에서 영감을 받은 | (파이프) 연산자를 사용
#   - A | B | C 의 형태로 코드를 작성하면, **"A의 출력 결과가 B의 입력으로 들어가고, B의 출력 결과가 C의 입력으로 들어간다"**는 의미
#     →  입력 데이터(Dictionary) | 프롬프트(Prompt) | AI 모델(LLM) | 출력 파서(Output Parser)




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



# 객체 생성 (본인의 직번과 키를 정확히 입력하세요)
llm = PgptLLM(
    api_key=os.getenv("API_KEY"),
    emp_no=os.getenv("EMP_NO"),
    model_name="gpt-4.1-nano",
    )
StreamResponse(llm.stream('LCEL : LangChain Expression Language 에 대해 자세히 알려줘.'))

chain = prompt_template | llm
chain.invoke({'country':'대한민국'})
chain.invoke('미국')


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
# Message 저장하고 불러오기
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ChatPromptTemplate.from_messages()를 사용해 역할별 메시지를 리스트로 구성합니다.
chat_prompt = ChatPromptTemplate.from_messages([
    # System: AI의 페르소나와 절대 규칙 부여
    ("system", """당신은 친절하고 전문적인 여행 가이드입니다. 
    다음 규칙을 반드시 지키세요:
    1. 전문 용어는 한국어로 번역하되, 괄호 안에 영어 원문을 병기할 것.
    2. 어조는 전문적이고 정중한 '~습니다' 체를 사용할 것.
    3. 사용자의 질문에 간결하게 답변할 것.
    """),
    
    # MessagesPlaceholder: 이전 대화 기록이 들어갈 자리 (변수명: chat_history)
    # optional=True :이 자리에 들어갈 데이터(변수)가 전달되지 않더라도, 에러를 뱉지 말고 그냥 빈칸으로 둔 채 넘어가라
    MessagesPlaceholder(variable_name="chat_history", optional=True),       
    
    # 4. User: 사용자의 실제 새로운 요청 (변수명: input_text)
    ("human", "{question}")
])


# LLM 및 체인 구성
chain = chat_prompt | llm

# ------------------------------------------------------------------------------
# 과거 대화 기록(Memory) 

chat_history = [
    HumanMessage(content="이번주 4월 22일부터 25일까지 일본 나가사키로 3박 4일 여행을 가려고 해."),
    AIMessage(content="나가사키 여행이시군요! 더 궁금한 점에 대해 질문해주세요.")
]

# question = "거기 지금 날씨가 어때? 옷차림을 어떻게 해야 할까?"
# question = "나가사키에 어린아이와 같이 여행가려고 하는데 나가사키 역 근처의 갈만한 장소 추천해줘. 온도와 날씨 고려해서."
# question = "방금 추천해준 두번째 장소에 대해 자세히 알려줘."
# question = "그 장소의 위치와 주소도 알려주고, 나가사키역에서 그 장소까지 어떻게 이동해야하는지도 알려줘."
response = StreamResponse(chain.stream({
        "chat_history": chat_history,
        "question": question
    }))

chat_history.append(HumanMessage(content=question))
chat_history.append(AIMessage(content=response.content))



# ------------------------------------------------------------------------------
# 과거 대화 기록(Json file) 
from langchain_community.chat_message_histories import FileChatMessageHistory
# 1. 파일 기반의 히스토리 객체 생성 (파일이 없으면 자동 생성됨)

chat_history = FileChatMessageHistory(f"{folder_path}/database/chat_history.json")
chat_history.messages       # messages 내용보기

# question = "이번주 4월 22일부터 25일까지 일본 나가사키로 3박 4일 여행을 가려고 해."
# question = "거기 지금 날씨가 어때? 옷차림을 어떻게 해야 할까?"
# question = "나가사키에 어린아이와 같이 여행가려고 하는데 나가사키 역 근처의 갈만한 장소 추천해줘. 온도와 날씨 고려해서."
# question = "방금 추천해준 두번째 장소에 대해 자세히 알려줘."
question = "그 장소의 위치와 주소도 알려주고, 나가사키역에서 그 장소까지 어떻게 이동해야하는지도 알려줘."

# Messages1 : 기본방법
history_messages = chat_history.messages 

# Messages2 최적화 : 최근 10개(human 5개, ai답변 5개)만 slicing하여 전달
history_messages = chat_history.messages[-10:] if len(chat_history.messages) > 10 else chat_history.messages

response = StreamResponse(chain.stream({
        "chat_history": history_messages,
        "question": question
    }))

# 새로운 메시지 추가 (추가하는 순간 json 파일에 자동 저장됨)
chat_history.add_user_message(question)
chat_history.add_ai_message(response.content)


# --------------------------------------------------------------------------------
# # (더 발전된 구현 방법) RunnableWithMessageHistory 사용: 세션 ID(session_id)를 기반으로 여러 사용자의 대화 기록을 쉽게 분리하여 관리
# from langchain_core.runnables.history import RunnableWithMessageHistory

# # 1. 세션 ID에 따라 히스토리를 불러오는 함수 정의
# def get_session_history(session_id: str):
#     return FileChatMessageHistory(f"{folder_path}/database/{session_id}_history.json")

# # 2. 체인에 히스토리 관리 기능 래핑
# chain_with_history = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key="question",
#     history_messages_key="chat_history",
# )

# # 3. 실행 (chat_history를 직접 넘길 필요 없이 session_id만 지정하면 자동 처리됨)
# response = StreamResponse(chain_with_history.stream(
#     {"question": question},
#     config={"configurable": {"session_id": "user_123"}} # 사용자 식별자
# ))
# --------------------------------------------------------------------------------


# # ----------------------------------------------------------------------
# # (참고) 실무에서 자주 쓰이는 자동화 패턴
# from langchain_core.runnables.history import RunnableWithMessageHistory

# # 체인을 RunnableWithMessageHistory로 감싸면, 
# # invoke 할 때 session_id만 넘겨주면 알아서 DB에서 과거 대화를 가져와 
# # MessagesPlaceholder 자리에 꽂아줍니다.
# with_message_history = RunnableWithMessageHistory(
#     chain,
#     get_session_history=get_chat_history_from_db, # DB 조회 함수
#     input_messages_key="question",
#     history_messages_key="chat_history",
# )
# # ----------------------------------------------------------------------