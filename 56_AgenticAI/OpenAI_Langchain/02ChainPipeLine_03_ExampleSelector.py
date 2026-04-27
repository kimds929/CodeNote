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





#########################################################################################################################
## FewShotChatMessagePromptTemplate

examples = [
    {
        "instruction": "당신은 회의록 작성 전문가 입니다. 주어진 정보를 바탕으로 회의록을 작성해 주세요",
        "input": "2023년 12월 25일, XYZ 회사의 마케팅 전략 회의가 오후 3시에 시작되었다. 회의에는 마케팅 팀장인 김수진, 디지털 마케팅 담당자인 박지민, 소셜 미디어 관리자인 이준호가 참석했다. 회의의 주요 목적은 2024년 상반기 마케팅 전략을 수립하고, 새로운 소셜 미디어 캠페인에 대한 아이디어를 논의하는 것이었다. 팀장인 김수진은 최근 시장 동향에 대한 간략한 개요를 제공했으며, 이어서 각 팀원이 자신의 분야에서의 전략적 아이디어를 발표했다.",
        "answer": """
회의록: XYZ 회사 마케팅 전략 회의
일시: 2023년 12월 25일
장소: XYZ 회사 회의실
참석자: 김수진 (마케팅 팀장), 박지민 (디지털 마케팅 담당자), 이준호 (소셜 미디어 관리자)

1. 개회
   - 회의는 김수진 팀장의 개회사로 시작됨.
   - 회의의 목적은 2024년 상반기 마케팅 전략 수립 및 새로운 소셜 미디어 캠페인 아이디어 논의.

2. 시장 동향 개요 (김수진)
   - 김수진 팀장은 최근 시장 동향에 대한 분석을 제시.
   - 소비자 행동 변화와 경쟁사 전략에 대한 통찰 공유.

3. 디지털 마케팅 전략 (박지민)
   - 박지민은 디지털 마케팅 전략에 대해 발표.
   - 온라인 광고와 SEO 최적화 방안에 중점을 둠.

4. 소셜 미디어 캠페인 (이준호)
   - 이준호는 새로운 소셜 미디어 캠페인에 대한 아이디어를 제안.
   - 인플루언서 마케팅과 콘텐츠 전략에 대한 계획을 설명함.

5. 종합 논의
   - 팀원들 간의 아이디어 공유 및 토론.
   - 각 전략에 대한 예산 및 자원 배분에 대해 논의.

6. 마무리
   - 다음 회의 날짜 및 시간 확정.
   - 회의록 정리 및 배포는 박지민 담당.
""",
    },
    {
        "instruction": "당신은 요약 전문가 입니다. 다음 주어진 정보를 바탕으로 내용을 요약해 주세요",
        "input": "이 문서는 '지속 가능한 도시 개발을 위한 전략'에 대한 20페이지 분량의 보고서입니다. 보고서는 지속 가능한 도시 개발의 중요성, 현재 도시화의 문제점, 그리고 도시 개발을 지속 가능하게 만들기 위한 다양한 전략을 포괄적으로 다루고 있습니다. 이 보고서는 또한 성공적인 지속 가능한 도시 개발 사례를 여러 국가에서 소개하고, 이러한 사례들을 통해 얻은 교훈을 요약하고 있습니다.",
        "answer": """
문서 요약: 지속 가능한 도시 개발을 위한 전략 보고서

- 중요성: 지속 가능한 도시 개발이 필수적인 이유와 그에 따른 사회적, 경제적, 환경적 이익을 강조.
- 현 문제점: 현재의 도시화 과정에서 발생하는 주요 문제점들, 예를 들어 환경 오염, 자원 고갈, 불평등 증가 등을 분석.
- 전략: 지속 가능한 도시 개발을 달성하기 위한 다양한 전략 제시. 이에는 친환경 건축, 대중교통 개선, 에너지 효율성 증대, 지역사회 참여 강화 등이 포함됨.
- 사례 연구: 전 세계 여러 도시의 성공적인 지속 가능한 개발 사례를 소개. 예를 들어, 덴마크의 코펜하겐, 일본의 요코하마 등의 사례를 통해 실현 가능한 전략들을 설명.
- 교훈: 이러한 사례들에서 얻은 주요 교훈을 요약. 강조된 교훈에는 다각적 접근의 중요성, 지역사회와의 협력, 장기적 계획의 필요성 등이 포함됨.

이 보고서는 지속 가능한 도시 개발이 어떻게 현실적이고 효과적인 형태로 이루어질 수 있는지에 대한 심도 있는 분석을 제공합니다.
""",
    },
    {
        "instruction": "당신은 문장 교정 전문가 입니다. 다음 주어진 문장을 교정해 주세요",
        "input": "우리 회사는 새로운 마케팅 전략을 도입하려고 한다. 이를 통해 고객과의 소통이 더 효과적이 될 것이다.",
        "answer": "본 회사는 새로운 마케팅 전략을 도입함으로써, 고객과의 소통을 보다 효과적으로 개선할 수 있을 것으로 기대된다.",
    },
]


from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.example_selectors import (
    SemanticSimilarityExampleSelector,
)
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

chroma = Chroma("fewshot_chat", OpenAIEmbeddings())

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{instruction}:\n{input}"),
        ("ai", "{answer}"),
    ]
)

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # 여기에는 선택 가능한 예시 목록이 있습니다.
    examples,
    # 여기에는 의미적 유사성을 측정하는 데 사용되는 임베딩을 생성하는 임베딩 클래스가 있습니다.
    OpenAIEmbeddings(),
    # 여기에는 임베딩을 저장하고 유사성 검색을 수행하는 데 사용되는 VectorStore 클래스가 있습니다.
    chroma,
    # 이것은 생성할 예시의 수입니다.
    k=1,
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
)


question = {
    "instruction": "회의록을 작성해 주세요",
    "input": "2023년 12월 26일, ABC 기술 회사의 제품 개발 팀은 새로운 모바일 애플리케이션 프로젝트에 대한 주간 진행 상황 회의를 가졌다. 이 회의에는 프로젝트 매니저인 최현수, 주요 개발자인 황지연, UI/UX 디자이너인 김태영이 참석했다. 회의의 주요 목적은 프로젝트의 현재 진행 상황을 검토하고, 다가오는 마일스톤에 대한 계획을 수립하는 것이었다. 각 팀원은 자신의 작업 영역에 대한 업데이트를 제공했고, 팀은 다음 주까지의 목표를 설정했다.",
}

example_selector.select_examples(question)

final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.",
        ),
        few_shot_prompt,
        ("human", "{instruction}\n{input}"),
    ]
)



# chain 생성
chain = final_prompt | llm

# 실행 및 결과 출력
answer = chain.stream(question)
result = StreamResponse(answer)


# ----------------------------------------------------------------------------------------------------------------
### Example Selector 의 유사도 검색 문제 해결
# 유사도 계산시 `instruction` 과 `input` 을 사용하고 있습니다. 하지만, `instruction` 만 사용하여 검색시 제대로된 유사도 결과가 나오지 않습니다. 
# 이를 해결하기 위해 커스텀 유사도 계산을 위한 클래스를 정의합니다.

question = {
    "instruction": "회의록을 작성해 주세요",
}

# 커스텀 하지 않은 기본 예제 선택기를 사용했을 때 결과
example_selector.select_examples({"instruction": "다음 문장을 요약해 주세요"})




from langchain_teddynote.prompts import CustomExampleSelector

# 커스텀 예제 선택기 생성
custom_selector = CustomExampleSelector(examples, OpenAIEmbeddings())

# 커스텀 예제 선택기를 사용했을 때 결과
custom_selector.select_examples({"instruction": "다음 문장을 회의록 작성해 주세요"})

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{instruction}:\n{input}"),
        ("ai", "{answer}"),
    ]
)

custom_fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=custom_selector,  # 커스텀 예제 선택기 사용
    example_prompt=example_prompt,  # 예제 프롬프트 사용
)

custom_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.",
        ),
        custom_fewshot_prompt,
        ("human", "{instruction}\n{input}"),
    ]
)

# chain 을 생성합니다.
chain = custom_prompt | llm

# 회의록 작성 Example ------------------------------------------------------------------------------
question = {
    "instruction": "회의록을 작성해 주세요",
    "input": "2023년 12월 26일, ABC 기술 회사의 제품 개발 팀은 새로운 모바일 애플리케이션 프로젝트에 대한 주간 진행 상황 회의를 가졌다. 이 회의에는 프로젝트 매니저인 최현수, 주요 개발자인 황지연, UI/UX 디자이너인 김태영이 참석했다. 회의의 주요 목적은 프로젝트의 현재 진행 상황을 검토하고, 다가오는 마일스톤에 대한 계획을 수립하는 것이었다. 각 팀원은 자신의 작업 영역에 대한 업데이트를 제공했고, 팀은 다음 주까지의 목표를 설정했다.",
}

# 실행 및 결과 출력
result = StreamResponse(chain.stream(question))


# 요약 작성 Example ------------------------------------------------------------------------------
question = {
    "instruction": "문서를 요약해 주세요",
    "input": "이 문서는 '2023년 글로벌 경제 전망'에 관한 30페이지에 달하는 상세한 보고서입니다. 보고서는 세계 경제의 현재 상태, 주요 국가들의 경제 성장률, 글로벌 무역 동향, 그리고 다가오는 해에 대한 경제 예측을 다룹니다. 이 보고서는 또한 다양한 경제적, 정치적, 환경적 요인들이 세계 경제에 미칠 영향을 분석하고 있습니다.",
}

# 실행 및 결과 출력
result = StreamResponse(chain.stream(question))


# 교정Example ------------------------------------------------------------------------------
question = {
    "instruction": "문장을 교정해 주세요",
    "input": "회사는 올해 매출이 증가할 것으로 예상한다. 새로운 전략이 잘 작동하고 있다.",
}

# 실행 및 결과 출력
result = StreamResponse(chain.stream(question))

