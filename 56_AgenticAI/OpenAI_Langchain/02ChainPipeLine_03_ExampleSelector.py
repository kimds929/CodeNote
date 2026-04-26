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

# ------------------------------------------------------------------------------------------------
# 【 ExampleSelector 】 예제가 많은 경우 프롬프트에 포함할 예제를 선택해야 할 수도 있습니다. 
#                       Example Selector 는 “많은 예시 중에서 지금 질문에 가장 적합한 몇 개만 골라주는 장치”다.
#  ✔️ 필요성
#       FewShotPromptTemplate만 쓰면 이런 문제가 생김:
#           예시를 고정하면 → 모든 상황에 똑같이 적용됨
#           예시를 많이 넣으면 → 토큰 낭비 + 성능 저하
#      👉 그래서 등장한 게 ExampleSelector
# ✔️ 작동 원리
#   👉 “현재 질문과 가장 비슷한 예시만 뽑자” (유사도 기반 검색 시스템)
#       1. 예시들을 벡터화 (임베딩)
#       2. 입력 질문도 벡터화
#       3. 유사도 계산 → 가장 유사한 예시 선택

# - [API 문서](https://api.python.langchain.com/en/latest/core/example_selectors.html)


from langchain_core.example_selectors import (
    SemanticSimilarityExampleSelector,      # 의미적으로 비슷한 예시 선택
    MaxMarginalRelevanceExampleSelector,    # 비슷하면서도 “서로 다른” 예시 선택
    LengthBasedExampleSelector,             # 길이에 기반한 예시 선택
    )
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma



examples = [
    {
        "question": "스티브 잡스와 아인슈타인 중 누가 더 오래 살았나요?",
        "answer": """이 질문에 추가 질문이 필요한가요: 예.
추가 질문: 스티브 잡스는 몇 살에 사망했나요?
중간 답변: 스티브 잡스는 56세에 사망했습니다.
추가 질문: 아인슈타인은 몇 살에 사망했나요?
중간 답변: 아인슈타인은 76세에 사망했습니다.
최종 답변은: 아인슈타인
""",
    },
    {
        "question": "네이버의 창립자는 언제 태어났나요?",
        "answer": """이 질문에 추가 질문이 필요한가요: 예.
추가 질문: 네이버의 창립자는 누구인가요?
중간 답변: 네이버는 이해진에 의해 창립되었습니다.
추가 질문: 이해진은 언제 태어났나요?
중간 답변: 이해진은 1967년 6월 22일에 태어났습니다.
최종 답변은: 1967년 6월 22일
""",
    },
    {
        "question": "율곡 이이의 어머니가 태어난 해의 통치하던 왕은 누구인가요?",
        "answer": """이 질문에 추가 질문이 필요한가요: 예.
추가 질문: 율곡 이이의 어머니는 누구인가요?
중간 답변: 율곡 이이의 어머니는 신사임당입니다.
추가 질문: 신사임당은 언제 태어났나요?
중간 답변: 신사임당은 1504년에 태어났습니다.
추가 질문: 1504년에 조선을 통치한 왕은 누구인가요?
중간 답변: 1504년에 조선을 통치한 왕은 연산군입니다.
최종 답변은: 연산군
""",
    },
    {
        "question": "올드보이와 기생충의 감독이 같은 나라 출신인가요?",
        "answer": """이 질문에 추가 질문이 필요한가요: 예.
추가 질문: 올드보이의 감독은 누구인가요?
중간 답변: 올드보이의 감독은 박찬욱입니다.
추가 질문: 박찬욱은 어느 나라 출신인가요?
중간 답변: 박찬욱은 대한민국 출신입니다.
추가 질문: 기생충의 감독은 누구인가요?
중간 답변: 기생충의 감독은 봉준호입니다.
추가 질문: 봉준호는 어느 나라 출신인가요?
중간 답변: 봉준호는 대한민국 출신입니다.
최종 답변은: 예
""",
    },
]

database_path = "D:/DataScience/DataBase"

#########################################################################################################################
from langchain_chroma import Chroma
# Chroma : 로컬 중심, 가볍게 쓰는 벡터 DB (개발/RAG용)
# * Milvus : 대규모, 고성능 벡터 DB (서비스용)
#

# 작동 원리
# [1] Embedding Layer → [2] Storage Layer → [3] Search Layer  
#   1. Embedding Layer : 텍스트를 벡터로 변환하는 역할을 합니다. (예: OpenAIEmbeddings)
#   2. Storage Layer : 벡터를 저장하는 역할을 합니다. (예: Chroma, Milvus)
#   3. Search Layer : 저장된 벡터에서 유사한 벡터를 검색하는 역할을 합니다. (예: Chroma의 유사도 검색 기능)


# Vector DB 생성 (저장소 이름, 임베딩 클래스) : 동일한 collection_name을 사용하면 이미 저장된 vectorDB에 접근할 수 있다.
chroma = Chroma(collection_name="example_selector", 
                embedding_function=OpenAIEmbeddings(),
                persist_directory=f"{database_path}/LLM_DB/chroma_db"   # 미지정시 RAM에만 저장(휘발성)
                )

data = chroma.get() # 저장된 벡터 검색
# {'ids': [],               # 저장된 벡터의 고유 ID 목록
#  'embeddings': None,      # 저장된 벡터의 임베딩 값 (벡터 데이터)
#  'documents': [],         # 저장된 벡터와 연관된 원본 문서 목록 (예시 데이터)
#  'uris': None,            # 저장된 벡터와 연관된 URI 목록 (예시 데이터의 출처나 위치 정보)
#  'data': None,            # 저장된 벡터와 연관된 추가 데이터 (예시 데이터와 관련된 메타데이터나 기타 정보)
#  'metadatas': [],         # 저장된 벡터와 연관된 메타데이터 목록 (예시 데이터와 관련된 추가 정보, 예: 생성 날짜, 작성자 등)
#  'included': [<IncludeEnum.documents: 'documents'>,   # 저장된 벡터와 연관된 원본 문서 목록이 포함됨
#               <IncludeEnum.uris: 'uris'>,             # 저장된 벡터와 연관된 URI 목록이 포함됨
#               <IncludeEnum.data: 'data'>,             # 저장된 벡터와 연관된 추가 데이터가 포함됨
#               <IncludeEnum.metadatas: 'metadatas'>]}  # 저장된 벡터와 연관된 메타데이터 목록이 포함됨

for doc, meta in zip(data["documents"], data["metadatas"]):
    print(doc, meta)
# chroma.similarity_search("강아지")        # 검색 기반 확인 (실전에서 더 많이 씀)





#########################################################################################################################

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples = examples,                    # 선택 가능한 예시 목록
    embeddings =  OpenAIEmbeddings(),       # 의미적 유사성을 측정하는 데 사용되는 임베딩을 생성하는 임베딩 클래스
    # vectorstore_cls = Chroma,             # 새로운 Chroma VectorDB를 생성(휘발성)
    vectorstore=chroma,                     # 이미 생성된 Chroma VectorDB를 사용하여 예시를 검색
    k = 1,                                  # 이것은 생성할 예시의 수입니다.
)

from langchain.schema import Document
[Document(page_content=ex["question"]) for ex in examples]
[Document(page_content=ex["answer"]) for ex in examples]

question = "Google이 창립된 연도에 Bill Gates의 나이는 몇 살인가요?"

# 입력과 가장 유사한 예시를 선택합니다.
selected_examples = example_selector.select_examples({"question": question})

print(f"입력에 가장 유사한 예시:\n{question}\n")
for example in selected_examples:
    print(f'question:\n{example["question"]}')
    print(f'answer:\n{example["answer"]}')
