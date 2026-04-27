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
##############################################################################################################
# OutputParser
##############################################################################################################
##############################################################################################################
# 원하는 Output을 완벽하게 얻기 위한 추가 고려사항 (실무 꿀팁)
#   . OutputParser만 붙인다고 해서 LLM이 알아서 완벽한 형태를 반환하지는 않습니다. 파서가 작동하려면 LLM이 파서가 해석할 수 있는 형태(예: 완벽한 JSON 포맷)로 텍스트를 뱉어내야 합니다. 이를 위해 다음 사항들을 반드시 고려해야 합니다.
#
#   ① 프롬프트에 형식 지시사항(Format Instructions) 명시
#       . 위의 예시 코드들에서 parser.get_format_instructions()를 프롬프트에 주입한 것을 보셨을 것입니다. 
#       이는 LLM에게 "반드시 이런 JSON 스키마로 대답해"라고 알려주는 역할을 합니다.
#       이 과정이 누락되면 파싱 에러가 발생할 확률이 매우 높습니다.
#
#   ② OpenAI의 with_structured_output() 활용 (가장 추천하는 최신 방식)
#       . 최근 LangChain과 OpenAI를 사용할 때, 복잡한 프롬프트 엔지니어링이나 OutputParser 없이도 가장 확실하게 Dictionary/Pydantic 객체를 얻는 방법입니다. 
#       OpenAI의 'Function Calling(Tool Calling)' 기능을 내부적으로 활용하여 모델 자체가 구조화된 데이터를 반환하도록 강제합니다.

# 주요 OutputParser
#   . 단순 텍스트: StrOutputParser
#   . 단어/키워드 목록: CommaSeparatedListOutputParser
#   . 문장/복잡한 목록: MarkdownListOutputParser
#   . 분류/카테고리 선택: EnumOutputParser
#   . 날짜/시간 추출: DatetimeOutputParser
#   . 동적/가벼운 데이터 구조: JsonOutputParser
#   . 엄격한 데이터 구조 및 API 연동 (가장 추천): PydanticOutputParser (또는 with_structured_output)
#   . 운영 환경 안정성 확보: OutputFixingParser로 기존 파서 감싸기


from langchain_core.prompts import ChatPromptTemplate

# ------------------------------------------------------------------------------------------------------------
# StrOutputParser
from langchain_core.output_parsers import StrOutputParser

prompt_template = ChatPromptTemplate.from_template(
    "{subject}와 관련된 핵심 키워드 5개를 추천해줘."
)

chain = prompt_template | llm | StrOutputParser()


response = chain.invoke({'subject': '인공지능'})
print(response)


# ------------------------------------------------------------------------------------------------------------
# PydanticOutputParser (고정된 형식의 Dictionary) : 엄격한 데이터 구조 및 API 연동
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 1. 원하는 데이터 구조를 Pydantic 클래스로 정의
class MarketingIdea(BaseModel):
    title: str = Field(description="마케팅 캠페인의 매력적인 제목")
    target_audience: list[str] = Field(description="타겟 고객층 목록")
    budget_estimate: int = Field(description="예상 예산 (원 단위 숫자만)")

# 방법 1 : PydanticOutputParser 활용
parser = PydanticOutputParser(pydantic_object=MarketingIdea)

prompt = ChatPromptTemplate.from_template(
    "다음 제품을 위한 마케팅 아이디어를 제안해줘: {product}\n\n{format_instructions}"
)

prompt_partial = prompt.partial(format_instructions=parser.get_format_instructions())

chain = prompt_partial | llm | parser

response = chain.invoke({
    "product": "무선 노이즈 캔슬링 이어폰"})
print(response)
print(response.title)           # 객체 속성으로 바로 접근 가능
print(response.model_dump())    # Dictionary 형태로 변환


# 방법 2 : PydanticOutputParser를 쓰는 대신, 모델 자체에 스키마를 바인딩합니다.
structured_llm = llm.with_structured_output(MarketingIdea)

prompt = ChatPromptTemplate.from_template("다음 제품을 위한 마케팅 아이디어를 제안해줘: {product}")
chain = prompt | structured_llm

response = chain.invoke({"product": "무선 노이즈 캔슬링 이어폰"})
print(response)
print(response.title)            # 객체 속성으로 바로 접근 가능
print(response.model_dump())     # Dictionary 형태로 변환



# Pydantic 클래스 정의 (Dict 타입 Return)
from typing import Dict, List
# 방법 1)
# class TravelRecommendations(BaseModel):
#     # Dict[str, str]을 통해 {문자열: 문자열} 형태임을 명시
#     destinations: Dict[str, str] = Field(
#         description="추천하는 '여행지 이름'을 키(Key)로, '추천 이유'를 값(Value)으로 가지는 딕셔너리"
#     )

# 방법 2) 고정된 키를 가진 객체들의 리스트를 생성할 때 오류(Hallucination)가 적다.
class Destination(BaseModel):
    name: str = Field(description="여행지 이름")
    reason: str = Field(description="추천 이유")

# 전체 응답 구조 (리스트 형태)
class TravelRecommendations(BaseModel):
    destinations: List[Destination] = Field(description="추천 여행지 목록")

parser = PydanticOutputParser(pydantic_object=TravelRecommendations)

prompt = ChatPromptTemplate.from_template(
    "넌 여행지를 추천하는 가이드야"
    "반드시 마크다운이나 다른 설명 없이 JSON 데이터만 출력해."  # LLM의 불필요한 말대꾸 방지
    "{question}\n\n{format_instructions}\n"
)

prompt_partial = prompt.partial(format_instructions=parser.get_format_instructions())

chain = prompt_partial | llm | parser

# 6. 실행
question = "한국 여행에서 꼭 가봐야할 여행지들을 3곳 추천해주고 그 이유를 알려줘."
response = chain.invoke({"question": question})
print(response)
print(response.destinations)



# ------------------------------------------------------------------------------------------------------------
# CommaSeparatedListOutputParser (List)
from langchain_core.output_parsers import CommaSeparatedListOutputParser

comma_sep_output_parser = CommaSeparatedListOutputParser()
format_instructions = comma_sep_output_parser.get_format_instructions()

prompt_template = ChatPromptTemplate.from_template(
    "{subject}와 관련된 핵심 키워드 5개를 추천해줘.\n{format_instructions}"
)

chain = prompt_template | llm | comma_sep_output_parser

response = chain.invoke({'subject': '인공지능',
                         'format_instructions':format_instructions})
print(response)


# ------------------------------------------------------------------------------------------------------------
# JsonOutputParser (비고정된 형식의 Dictionary)
from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
prompt = ChatPromptTemplate.from_template(
    "다음 문장에서 등장인물의 이름과 나이를 추출해줘.\n{format_instructions}\n문장: {text}"
)
chain = prompt | llm | parser

response = chain.invoke({
    "text": "철수는 25살이고, 영희는 23살이야.",
    "format_instructions": parser.get_format_instructions()
})
print(response)
# 출력(Dict): {'철수': 25, '영희': 23}


# ------------------------------------------------------------------------------------------------------------
# DatetimeOutputParser : 날짜 및 시간 형태 (Datetime) 
from langchain.output_parsers import DatetimeOutputParser
parser = DatetimeOutputParser()
prompt = ChatPromptTemplate.from_template(
    "사용자의 요청에서 날짜와 시간을 추출해.\n{format_instructions}\n요청: {request}"
)
chain = prompt | llm | parser

response = chain.invoke({
    "request": "2024년 12월 25일 저녁 6시에 알람 맞춰줘.",
    "format_instructions": parser.get_format_instructions()
})
print(response)     # 출력(datetime 객체): 2024-12-25 18:00:00





# ------------------------------------------------------------------------------------------------------------
#  EnumOutputParser : 객관식 선택 형태 (Enum) - 
from enum import Enum
from langchain.output_parsers import EnumOutputParser

class Sentiment(Enum):
    POSITIVE = "긍정"
    NEGATIVE = "부정"
    NEUTRAL = "중립"

parser = EnumOutputParser(enum=Sentiment)
prompt = ChatPromptTemplate.from_template(
    "다음 리뷰의 감성을 분류해.\n{format_instructions}\n리뷰: {review}"
)
chain = prompt | llm | parser

response = chain.invoke({
    "review": "배송은 느렸지만 제품은 정말 마음에 들어요!",
    "format_instructions": parser.get_format_instructions()
})
print(response)

# ------------------------------------------------------------------------------------------------------------

# OutputFixingParser : 형태를 변환하는 것이 아니라 "고쳐주는" 특수 파서
from langchain.output_parsers import OutputFixingParser

# 기존에 쓰던 Pydantic 파서가 있다고 가정
base_parser = PydanticOutputParser(pydantic_object=MarketingIdea)

# FixingParser로 한 번 감싸줍니다.
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)






# ------------------------------------------------------------------------------------------------------------
# PandasDataFrameOutputParser : Pandas DataFrame 형태로 출력받기
from langchain.output_parsers import PandasDataFrameOutputParser
from langchain_core.prompts import PromptTemplate

import pprint
from typing import Any, Dict
import pandas as pd


# 출력 목적으로만 사용됩니다.
def format_parser_output(parser_output: Dict[str, Any]) -> None:
    # 파서 출력의 키들을 순회합니다.
    for key in parser_output.keys():
        # 각 키의 값을 딕셔너리로 변환합니다.
        parser_output[key] = parser_output[key].to_dict()
    # 예쁘게 출력합니다.
    return pprint.PrettyPrinter(width=4, compact=True).pprint(parser_output)


data_url = 'https://raw.githubusercontent.com/kimds929/CodeNote/refs/heads/main/99_DataSet/Data_Tabular/'

# 원하는 Pandas DataFrame을 정의합니다.
df = pd.read_csv(f"{data_url}/titanic_original.csv")
print(df.shape)



# 파서를 설정하고 프롬프트 템플릿에 지시사항을 주입합니다.
parser = PandasDataFrameOutputParser(dataframe=df)

# 파서의 지시사항을 출력합니다.
print(parser.get_format_instructions())


# 프롬프트 템플릿을 설정합니다.
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{question}\n",
    input_variables=["question"],  # 입력 변수 설정
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },  # 부분 변수 설정
)

# 체인 생성
chain = prompt | llm | parser


# Column조회 예시
df_query = "Age column 을 조회해 주세요."
parser_output = chain.invoke({"question": df_query})
format_parser_output(parser_output)


# 행 조회 예시
df_query = "Retrieve the first row."
parser_output = chain.invoke({"question": df_query})
format_parser_output(parser_output)




# 임의의 Pandas DataFrame 작업 예시, 행의 수를 제한합니다.
df_query = "Retrieve the average of the Ages from row 0 to 4."
parser_output = chain.invoke({"question": df_query})
print(parser_output)

df["age"].head().mean() # answer check


