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







##############################################################################################################
# ------------------------------------------------------------------------------------------------


from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. 사용자 정의 함수 정의 (RunnableLambda 용)
# ==========================================

# 가상의 외부 API 호출 함수 1: 시장 동향 가져오기
def fetch_market_trend(inputs: dict) -> str:
    company = inputs["company"]
    # 실제로는 여기서 뉴스 API나 검색 API를 호출합니다.
    print(f"[Log] '{company}'의 시장 동향 수집 중...")
    return f"{company}는 최근 AI 및 자동화 기술 도입에 막대한 투자를 진행 중이며, 친환경 트렌드에 맞춰 사업을 확장하고 있습니다."

# 가상의 외부 API 호출 함수 2: 경쟁사 정보 가져오기
def fetch_competitors(inputs: dict) -> list:
    company = inputs["company"]
    # 실제로는 DB 조회나 외부 API를 호출합니다.
    print(f"[Log] '{company}'의 경쟁사 정보 수집 중...")
    if company == "테슬라":
        return ["BYD", "현대자동차", "폭스바겐"]
    return ["경쟁사A", "경쟁사B"]

# 데이터 가공 함수: 병렬로 수집된 데이터를 프롬프트에 넣기 좋게 문자열로 포맷팅
def format_context(inputs: dict) -> str:
    trend = inputs["trend"]
    competitors = ", ".join(inputs["competitors"])
    return f"▶ 시장 동향: {trend}\n▶ 주요 경쟁사: {competitors}"


# ==========================================
# 2. 프롬프트 및 LLM 설정
# ==========================================

prompt = PromptTemplate.from_template(
    """당신은 최고 수준의 기업 전략 분석가입니다.
아래 제공된 기업명과 배경 정보를 바탕으로 향후 1년 전략 보고서를 작성해주세요.

[기업명]: {company}

[배경 정보]
{context}

[요구사항]
1. 배경 정보를 바탕으로 한 현재 상황 요약
2. 주요 경쟁사를 이기기 위한 차별화 전략 3가지
3. 결론 및 제언
"""
)

trend = fetch_market_trend({"company": "테슬라"})
print(trend)
competitors = fetch_competitors({"company": "테슬라"})
print(competitors)

format_context({"trend": trend, 'competitors': competitors})

# ==========================================
# 3. 체인(Chain) 구성 (핵심 파트)
# ==========================================

# 3-1. 정보 수집 및 가공 체인 (Parallel + Lambda)
# 입력받은 딕셔너리({"company": "..."})를 두 함수에 동시에 전달하여 실행합니다.
context_gathering_chain = (
    RunnableParallel(
        trend=RunnableLambda(fetch_market_trend),
        competitors=RunnableLambda(fetch_competitors)
    )
    | RunnableLambda(format_context) # 병렬 실행 결과를 받아 하나의 문자열로 가공
)

# 3-2. 메인 체인 조립 (Passthrough + Prompt + LLM)
chain = (
    # 기존 입력({"company": "테슬라"})을 유지하면서, 'context'라는 새로운 키에 수집된 정보를 할당합니다.
    RunnablePassthrough.assign(context = context_gathering_chain)
    | prompt
    | llm
    | StrOutputParser()
)

# 4. 체인 실행
input_data = {"company": "테슬라"}
result = chain.invoke(input_data)

print("\n=== [최종 생성된 보고서] ===")
print(result)


# RunnablePassthrough.assign(context = context_gathering_chain).invoke(input_data)
# context_gathering_chain.invoke(input_data)





