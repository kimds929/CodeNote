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

chain = prompt | llm | parser

response = chain.invoke({
    "product": "무선 노이즈 캔슬링 이어폰",
    "format_instructions": parser.get_format_instructions()
})
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


