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