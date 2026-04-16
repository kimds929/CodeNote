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

llm = ChatOpenAI(
    # temperature=0.1,  # 창의성 (0.0 ~ 2.0)
    # model_name="gpt-4o-mini",  # 모델명
    model_name="gpt-4.1-nano",  # 모델명
    # model_name="gpt-5-nano",  # 모델명
)

##############################################################################################################
# 이미지 입력

from langchain_teddynote.models import MultiModal
from langchain_teddynote.messages import stream_response


# 멀티모달 객체 생성
multimodal_llm = MultiModal(llm)


# 샘플 이미지 주소(웹사이트로 부터 바로 인식)
IMAGE_URL = "https://t3.ftcdn.net/jpg/03/77/33/96/360_F_377339633_Rtv9I77sSmSNcev8bEcnVxTHrXB4nRJ5.jpg"

# 이미지 파일로 부터 질의
# answer = multimodal_llm.invoke(IMAGE_URL)
answer = multimodal_llm.stream(IMAGE_URL)     # return : generator

# 스트리밍 방식으로 각 토큰을 출력합니다. (실시간 출력)
stream_response(answer)



# 로컬 PC 에 저장되어 있는 이미지의 경로 입력
IMAGE_PATH_FROM_FILE = r"D:\DataScience\Code9) Python\FastCampus_Teddy_Langchain_RAG\langchain-kr\images\langchain-note.png"

# 이미지 파일로 부터 질의(스트림 방식)
answer = multimodal_llm.stream(IMAGE_PATH_FROM_FILE)
# 스트리밍 방식으로 각 토큰을 출력합니다. (실시간 출력)
stream_response(answer)



## System, User 프롬프트 수정
system_prompt = """당신은 표(재무제표) 를 해석하는 금융 AI 어시스턴트 입니다. 
당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는 것입니다."""

user_prompt = """당신에게 주어진 표는 회사의 재무제표 입니다. 흥미로운 사실을 정리하여 답변하세요."""

# 멀티모달 객체 생성
multimodal_llm_with_prompt = MultiModal(
    llm, system_prompt=system_prompt, user_prompt=user_prompt
)
# 로컬 PC 에 저장되어 있는 이미지의 경로 입력
IMAGE_PATH_FROM_FILE = "https://storage.googleapis.com/static.fastcampus.co.kr/prod/uploads/202212/080345-661/kwon-01.png"

# 이미지 파일로 부터 질의(스트림 방식)
answer = multimodal_llm_with_prompt.stream(IMAGE_PATH_FROM_FILE)

# 스트리밍 방식으로 각 토큰을 출력합니다. (실시간 출력)
stream_response(answer)