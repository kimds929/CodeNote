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
        # model_name="gpt-4.1-nano",
        model_name="gpt-5-nano",
        # temperature=2.0,  # 정상 코드에 있던 설정값 적용
        # top_p=0.9,
        # stream_usage=True
    )
except:
    llm = ChatOpenAI(
        # temperature=0.1,  # 창의성 (0.0 ~ 2.0)
        # model_name="gpt-4o-mini",  # 모델명
        # model_name="gpt-4.1-nano",  # 모델명
        model_name="gpt-5-nano",  # 모델명
    )

    logging.langsmith("Default project")      # LangSmith 추적을 시작합니다.
    # logging.langsmith("Default project", set_enable=False)  # LangSmith 추적을 하지 않습니다.
print(llm)

##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################








##############################################################################################################
##############################################################################################################
# Message 저장하고 불러오기 (대화내용 기록)
##############################################################################################################
##############################################################################################################
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

question = "거기 지금 날씨가 어때? 옷차림을 어떻게 해야 할까?"
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

question = "이번주 4월 22일부터 25일까지 일본 나가사키로 3박 4일 여행을 가려고 해."
# question = "거기 지금 날씨가 어때? 옷차림을 어떻게 해야 할까?"
# question = "나가사키에 어린아이와 같이 여행가려고 하는데 나가사키 역 근처의 갈만한 장소 추천해줘. 온도와 날씨 고려해서."
question = "방금 추천해준 두번째 장소에 대해 자세히 알려줘."
# question = "그 장소의 위치와 주소도 알려주고, 나가사키역에서 그 장소까지 어떻게 이동해야하는지도 알려줘."

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





# ------------------------------------------------------------------------------
# 과거 대화 기록(DB: SQLite) 
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. SQLite DB에 연결하고 특정 세션(사용자 또는 채팅방)의 히스토리 객체 생성
session_id = "user_123_chat_001"
db_path = f"sqlite:///{folder_path}/database/chat_history.db"


chat_history_db = SQLChatMessageHistory(
    session_id=session_id,
    connection_string=db_path
)

# DB에 저장된 내용 확인
chat_history_db.messages

# 해당 session_id의 모든 대화 기록을 DB에서 삭제
chat_history_db.clear()


# Messages1 : 기본방법
history_messages = chat_history_db.messages 

# # Messages2 최적화 : 최근 10개(human 5개, ai답변 5개)만 slicing하여 전달
# history_messages = chat_history.messages[-10:] if len(chat_history.messages) > 10 else chat_history.messages
history_messages

# question = "이번주 4월 22일부터 25일까지 일본 나가사키로 3박 4일 여행을 가려고 해."
# question = "거기 지금 날씨가 어때? 옷차림을 어떻게 해야 할까?"
# question = "나가사키에 어린아이와 같이 여행가려고 하는데 나가사키 역 근처의 갈만한 장소 추천해줘. 온도와 날씨 고려해서."
question = "방금 추천해준 두번째 장소에 대해 자세히 알려줘."
# question = "그 장소의 위치와 주소도 알려주고, 나가사키역에서 그 장소까지 어떻게 이동해야하는지도 알려줘."

# response와 동시에 자동으로 저장됨
response = StreamResponse(chain.stream({
        "chat_history": history_messages,
        "question": question
        })
    )      # chat_history 정보가 config 변수를 통해 자동으로 입력됨


# # # 새로운 메시지 추가 (추가하는 순간 .db 파일에 자동 저장됨)
chat_history_db.add_user_message(question)
chat_history_db.add_ai_message(response.content)

##############################################################################################################




