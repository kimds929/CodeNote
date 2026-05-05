import os
current_file_path = os.path.abspath(__file__).replace('\\','/')

if os.path.isdir("D:/DataScience/★GitHub_kimds929"):
    start_script_folder ="D:/DataScience/★GitHub_kimds929/CodeNote/56_AgenticAI"
elif os.path.isdir("D:/DataScience/PythonforWork"):
    start_script_folder ="D:/DataScience/PythonForwork/AgenticAI"
elif os.path.isdir("C:/Users/kimds929/DataScience"):
    start_script_folder = "C:/Users/kimds929/DataScience/AgenticAI"
        
start_script_path = f'{start_script_folder}/StartingScript_AgenticAI.txt'

with open(start_script_path, 'r', encoding='utf-8') as f:
    script = f.read()
script_formatted = script.replace('{base_folder_name}', 'DataScience') \
                         .replace('{current_path}', current_file_path)
# print(script_formatted)
exec(script_formatted)

################################################################################################





################################################################################################
################################################################################################
################################################################################################
#
# 1) LangChain / RAG에서 Text Splitter란?                                                                            
#    **Text Splitter(텍스트 분할기)**는 긴 문서(원문)를 작은 조각(Chunk) 으로 나눠서 RAG 파이프라인에서 다루기 쉽게     
#    만드는 컴포넌트입니다.                                                                                             
#    LangChain에서는 주로 Document(page_content + metadata)을 입력받아 여러 개의 Document 조각들로 반환합니다.          
#
# -------------------------------------------------------------------------------------------------------------------
# 
# 2) 왜 필요한가? (RAG에서의 필요성)                                                                                 
# 
#     (1) 검색 품질(Recall/Precision) 개선                                                                               
#     • 임베딩 기반 검색은 “문서 전체”가 아니라 “의미 있는 단위”로 쪼개져 있을 때 관련 부분을 더 잘 찾습니다.           
#     • 너무 길면 여러 주제가 섞여 임베딩이 흐려지고(토픽 혼합), 검색 정확도가 떨어집니다.                              
#     • 너무 짧으면 문맥이 부족해 답변에 필요한 근거가 잘리지거나, 검색 결과가 산발적으로 나옵니다.                     
# 
#     (2) 모델 컨텍스트 한도/비용 문제                                                                                   
#     • LLM 입력 토큰 한도가 있으므로, 원문을 그대로 넣을 수 없는 경우가 많습니다.                                      
#     • Chunk 단위로 넣으면 필요한 부분만 컨텍스트로 구성할 수 있어 비용/지연도 감소합니다.                             
# 
#     (3) 후처리(출처, 하이라이팅, 인용) 용이                                                                            
#     • Chunk마다 metadata(문서명, 섹션, 페이지, 헤더 등)를 달아두면,                                                   
#         • 답변에 출처 표기                                                                                             
#         • 어떤 섹션에서 근거를 가져왔는지 추적                                                                         
#         • UI에서 하이라이트 표시 가 쉬워집니다.                                                                        
# 
#     (4) 중첩(overlap)으로 문맥 보존                                                                                    
#     • Chunk 경계에서 문장이 끊기면 이해가 어려울 수 있습니다.                                                         
#     • chunk_overlap을 주어 앞뒤 문맥을 일부 중복시켜 답변 품질을 올립니다.                                            
# 
# -------------------------------------------------------------------------------------------------------------------
# 
# 3) RAG에서 어떻게 사용되나? (일반 흐름)                                                                            
#      1 **로더(Loader)**로 문서 로드 (PDF/HTML/Markdown/JSON 등) → Document[]                                           
#      2 Text Splitter로 분할 → Document[] (chunk들)                                                                     
#      3 임베딩 생성(각 chunk) → 벡터 저장소(VectorStore)에 저장                                                         
#      4 질의 시:                                                                                                        
#         • 질의 임베딩 → 유사 chunk 검색                                                                                
#         • 검색된 chunk를 LLM 컨텍스트로 넣어 답변 생성                                                                 
# 
# -------------------------------------------------------------------------------------------------------------------
# 
# 4) Chunk 설계 시 핵심 파라미터                                                                                     
#     • chunk_size: chunk 목표 크기(문자/토큰 등 기준은 splitter 종류에 따라 다름)                                      
#     • chunk_overlap: chunk 간 중복 크기                                                                               
#     • separators: 어디서 끊을지(문단, 문장, 공백 등)                                                                  
#     • “좋은 값”은 데이터/모델/도메인에 따라 달라서 실험이 필요하지만,                                                 
#         • 일반 문서 RAG: 500~1500 토큰 정도 범위를 자주 씁니다(모델과 검색 방식에 따라 조정).                          
#         • overlap은 보통 5~20% 수준을 많이 사용합니다.          
# 
################################################################################################





################################################################################################
# A. CharacterTextSplitter    
#   지정된 단일 문자(기본값 \n\n)를 기준으로 텍스트를 단순하게 자릅니다.
#                                           
#    개념                                                                                                               
#     • 문자 수 기준으로 텍스트를 자릅니다.                                                                             
#     • 보통 separator(예: \n\n) 기준으로 먼저 나누고, 그 조각들이 chunk_size를 넘으면 더 자르는 식으로                 
#       동작합니다(구현/설정에 따라 다름).                                                                              
#
#    쓰임새                                                                                                             
#     • 구조가 단순한 plain text 문서                                                                                   
#     • “대충 일정 길이로 자르고 싶다”는 목적에 빠르게 적용 가능                                                        
#
#    장점                                                                                                               
#     • 빠르고 단순함                                                                                                   
#     • 토크나이저 의존이 적음                                                                                          
#
#    단점/주의                                                                                                          
#     • 문자 수는 실제 LLM 토큰 수와 정확히 대응하지 않음(언어/기호/이모지/공백 등에 따라 토큰 수 변동)                 
#     • 문장 경계를 보장하지 않으면 문맥이 끊길 수 있음             
# -------------------------------------------------------------------------------------------------------------------
 
from langchain_text_splitters import CharacterTextSplitter

text = "첫 번째 문단입니다.\n\n두 번째 문단입니다.\n\n세 번째 문단입니다."

# 구분자를 '\n\n'으로 설정하고, 청크 크기를 15로 제한
splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=15,
    chunk_overlap=0
)

chunks = splitter.split_text(text)
print(chunks)
# 출력: ['첫 번째 문단입니다.', '두 번째 문단입니다.', '세 번째 문단입니다.']





################################################################################################
#  B. RecursiveCharacterTextSplitter
#     문단(\n\n) ➔ 문장(\n) ➔ 단어( ) ➔ 글자("") 순서로 재귀적으로 자릅니다. 문맥을 가장 잘 보존합니다.
#                                   
#     개념                                                                                                               
#     • 가장 널리 쓰이는 방식 중 하나.                                                                                  
#     • “큰 구분자 → 작은 구분자” 순서로 재귀적으로 쪼개며, 가능한 한 “자연스러운 경계(문단/문장/단어)”를 유지하려고    
#     합니다.                                                                                                         
#     • 예: ["\n\n", "\n", " ", ""] 같은 separators를 두고,                                                             
#         • 문단으로 먼저 자르고,                                                                                        
#         • 너무 크면 줄바꿈,                                                                                            
#         • 또 크면 공백,                                                                                                
#         • 그래도 크면 문자 단위로 자릅니다.                                                                            
#
#     쓰임새                                                                                                             
#     • 문서 형식이 제각각인 범용 RAG 기본값으로 많이 사용                                                              
#     • PDF 추출 텍스트처럼 줄바꿈/공백이 애매한 경우에도 그나마 안정적                                                 
#
#     장점                                                                                                               
#     • chunk가 “자연스러운 단위”가 되기 쉬워 검색/요약/답변 품질이 안정적                                              
#     • 설정만 잘하면 대부분 문서에서 무난                                                                              
#
#     단점/주의                                                                                                          
#     • separators 설계를 잘못하면 오히려 이상한 단위로 쪼개질 수 있음                                                  
#     • 여전히 “토큰 단위 제어”는 아님     

# -------------------------------------------------------------------------------------------------------------------

from langchain_text_splitters import RecursiveCharacterTextSplitter

text = "랭체인은 LLM 애플리케이션 개발을 돕는 프레임워크입니다. RAG 시스템을 구축할 때 매우 유용하죠. 텍스트 스플리터는 필수입니다."

# 청크 크기를 30글자로, 겹치는 구간을 5글자로 설정
splitter = RecursiveCharacterTextSplitter(
    chunk_size=30,
    chunk_overlap=5
)

chunks = splitter.split_text(text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
    
# 출력:
# Chunk 1: 랭체인은 LLM 애플리케이션 개발을 돕는 프레임워크입니다.
# Chunk 2: 워크입니다. RAG 시스템을 구축할 때 매우 유용하죠.
# Chunk 3: 유용하죠. 텍스트 스플리터는 필수입니다.






################################################################################################
#  C. TokenTextSplitter                                                
#    글자 수가 아닌 LLM이 과금하는 단위인 '토큰(Token)' 기준으로 자릅니다.
#    
#    개념                                                                                                               
#     • 토큰 수 기준으로 분할합니다.                                                                                    
#     • 특정 토크나이저(OpenAI tiktoken 등)에 맞춰 “LLM 입력 크기”를 더 정확히 맞출 수 있습니다.                        
#    
#    쓰임새                                                                                                             
#     • 컨텍스트 한도/비용을 엄격히 관리해야 하는 서비스                                                                
#     • “chunk_size를 800 tokens로 고정”처럼 토큰 기반 운영이 필요한 경우                                               
#    
#    장점                                                                                                               
#     • 모델 입력 제한과 가장 직접적으로 맞물림(예측 가능)                                                              
#     • 다국어/특수문자에서도 문자 기반보다 일관적                                                                      
#    
#    단점/주의                                                                                                          
#     • 문장/문단 경계가 깨질 수 있음(토큰 기준으로 뚝 자름)                                                            
#     • 토크나이저 의존성이 생김(모델 변경 시 결과 달라질 수 있음)         
#    
# -------------------------------------------------------------------------------------------------------------------
from langchain_text_splitters import TokenTextSplitter

text = "Token splitters are very strict about the token limits."

# 토큰 5개 단위로 자르기 (tiktoken 라이브러리 필요)
splitter = TokenTextSplitter(
    chunk_size=5, 
    chunk_overlap=1
)

chunks = splitter.split_text(text)
print(chunks)
# 출력: ['Token splitters are very strict', 'strict about the token limits', 'limits.']




################################################################################################
#  D. SemanticChunker         
#   의미가 비슷한 문장끼리 묶어서 자릅니다. (임베딩 모델이 필요합니다)                                        
# 
#   개념                                                                                                               
#    • 의미(semantic) 변화 지점을 기준으로 chunk를 나눕니다.                                                           
#    • 보통 문장을 나열해 임베딩을 만들고, 인접 문장 간 유사도/거리 변화가 큰 지점에서 경계를 잡는 식입니다(구현체마다 
#      다름).                                                                                                          
#   
#   쓰임새                                                                                                             
#    • 한 문서 안에 토픽 전환이 잦고, “길이”보다 의미적 응집도가 중요한 콘텐츠:                                        
#       • 블로그/가이드/기술 문서                                                                                      
#       • 회의록, Q&A 모음                                                                                             
#    • “문단이 길거나 형식이 일정치 않아서 단순 분할이 성능이 안 나오는 경우”                                          
#   
#   장점                                                                                                               
#    • chunk가 한 가지 주제에 더 응집되기 쉬워 검색 정확도가 좋아질 수 있음                                            
#    • 불필요한 토픽 혼합을 줄임                                                                                       
#   
#   단점/주의                                                                                                          
#    • 임베딩 계산이 추가로 필요해 전처리 비용 증가                                                                    
#    • 파라미터(문장 단위 분리, 유사도 임계값 등) 튜닝이 필요                                                          
#    • 항상 좋은 건 아니고(특히 매우 짧은 텍스트/형식적 문서) 오히려 불안정할 수 있음                                  
# 
# -------------------------------------------------------------------------------------------------------------------

# embeddings.model_name = 'text-embedding-ada-002'
embeddings.model_name = 'text-embedding-3-small'
# embeddings.model_name = 'text-embedding-3-large'

from langchain_experimental.text_splitter import SemanticChunker

# text = "사과는 맛있다. 바나나도 달콤하다. (과일 이야기 끝) 자동차 엔진은 복잡하다. 타이어 교체 시기가 왔다."
text = "Apple is yammy. Banana is also delicious. (The end of fruit story) Structures of engine in Car are complicated. It is time to change my tire."

# OpenAI 임베딩 모델을 사용하여 의미 기반으로 분할
splitter = SemanticChunker(embeddings)

chunks = splitter.split_text(text)
print(chunks)
# 출력: 
# ['사과는 맛있다. 바나나도 달콤하다. (과일 이야기 끝)', 
#  '자동차 엔진은 복잡하다. 타이어 교체 시기가 왔다.']




################################################################################################
# E. MarkdownHeaderTextSplitter                                           
#   마크다운의 헤더(#)를 기준으로 자르고, 헤더 정보를 메타데이터로 저장합니다
# 
#    개념                                                                                                               
#     • Markdown의 헤더(#, ##, ### …)를 기준으로 섹션을 나눕니다.                                                       
#     • 헤더 구조를 metadata로 유지할 수 있어,                                                                          
#        • chunk가 “어느 섹션에 속하는지” 추적이 쉬워집니다.                                                            
#
#    쓰임새                                                                                                             
#     • README, 위키, 문서 사이트 원본이 Markdown인 경우                                                                
#     • “섹션 단위 검색/인용”이 중요한 경우                                                                             
#
#    장점                                                                                                               
#     • 문서 구조를 보존(검색 결과에 “어떤 장/절인지” 붙일 수 있음)                                                     
#     • 섹션 단위로 자연스럽게 나뉘어 품질이 안정적                                                                     
#
#    단점/주의                                                                                                          
#     • 헤더 없이 긴 본문이 이어지면 추가 분할이 필요(대개 header split 후 다른 splitter로 2차 분할)                    
#     • Markdown이 깨진 텍스트(변환 과정에서 헤더 누락)엔 효과 감소        
#
# -------------------------------------------------------------------------------------------------------------------
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """
# 1장. RAG 개요
RAG는 검색 증강 생성입니다.
## 1.1 장점
환각 현상을 줄여줍니다.
"""

# 어떤 헤더를 기준으로 자를지 정의
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
splits = splitter.split_text(markdown_text)

for split in splits:
    print(f"내용: {split.page_content}")
    print(f"메타데이터: {split.metadata}\n")

# 출력:
# 내용: RAG는 검색 증강 생성입니다.
# 메타데이터: {'Header 1': '1장. RAG 개요'}
#
# 내용: 환각 현상을 줄여줍니다.
# 메타데이터: {'Header 1': '1장. RAG 개요', 'Header 2': '1.1 장점'}




################################################################################################
# F. HTMLHeaderTextSplitter                                             
#    웹페이지의 HTML 태그(<h1>, <h2>)를 기준으로 자릅니다.
#
#    개념                                                                                                               
#     • HTML의 헤더 태그(<h1>~<h6>)를 기준으로 섹션을 나눕니다.                                                         
#     • 웹페이지/크롤링 결과에 적합하며, 섹션 구조를 metadata로 담을 수 있습니다.                                       
#
#    쓰임새                                                                                                             
#     • 웹 문서(기술 문서, 블로그, 정책 문서)를 크롤링해서 RAG 구성할 때                                                
#     • 메뉴/푸터/광고 제거 후 본문을 구조적으로 분해하고 싶을 때                                                       
#
#    장점                                                                                                               
#     • 웹 문서의 논리적 구조(heading hierarchy)를 활용                                                                 
#     • 섹션별 출처/제목을 유지하기 쉬움                                                                                
#
#    단점/주의                                                                                                          
#     • HTML이 지저분하면(헤더 태그 남용/누락) 결과가 불안정                                                            
#     • 본문 추출(boilerplate 제거) 단계가 품질을 크게 좌우                  
#
# -------------------------------------------------------------------------------------------------------------------
from langchain_text_splitters import HTMLHeaderTextSplitter

html_text = """
<html>
    <body>
        <h1>회사 소개</h1>
        <p>우리는 AI 회사입니다.</p>
        <h2>연혁</h2>
        <p>2023년 설립되었습니다.</p>
    </body>
</html>
"""

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
splits = splitter.split_text(html_text)

for split in splits:
    print(f"메타데이터: {split.metadata} | 내용: {split.page_content}")

# 출력:
# 메타데이터: {'Header 1': '회사 소개'} | 내용: 우리는 AI 회사입니다.
# 메타데이터: {'Header 1': '회사 소개', 'Header 2': '연혁'} | 내용: 2023년 설립되었습니다.






################################################################################################
# G. RecursiveJsonSplitter                                              
#     JSON 데이터의 괄호 구조가 깨지지 않게 안전하게 자릅니다.
# 
# 개념                                                                                                               
#  • JSON을 “키-값 구조”를 유지하면서 분할합니다.                                                                    
#  • 일반적인 텍스트 split이 아니라, 중첩 객체/배열을 재귀적으로 탐색하면서 크기 제한에 맞게 조각을 나눕니다.        
#  • chunk에는 “어떤 경로(path)의 데이터인지” 같은 구조 정보가 함께 가는 경우가 많습니다.                            
# 
# 쓰임새                                                                                                             
#  • 로그, 이벤트, 설정, 스키마, API 응답, 제품 카탈로그처럼 구조화 데이터(JSON) 를 RAG에 넣어야 할 때               
#  • “필드 단위 검색/근거 제시”가 필요한 경우                                                                        
# 
# 장점                                                                                                               
#  • JSON 구조를 보존해서 “문맥”이 텍스트보다 명확함                                                                 
#  • 특정 키/경로 기반의 추적이 가능해 디버깅/출처 표기에 유리                                                       
# 
# 단점/주의                                                                                                          
#  • 단순 텍스트 임베딩만으로는 “정확한 필드 매칭”이 부족할 수 있어,                                                 
#     • (1) key를 텍스트로 잘 직렬화(serialization)하거나                                                            
#     • (2) 하이브리드 검색(BM25+vector),                                                                            
#     • (3) 필드별 인덱싱 전략 등을 함께 고려하는 게 좋습니다.                                                       
# 
# -------------------------------------------------------------------------------------------------------------------

from langchain_text_splitters import RecursiveJsonSplitter
import json

json_data = {
    "company": "AI Corp",
    "employees": [
        {"name": "Alice", "role": "Developer", "skills": ["Python", "C++", "Java", "Go", "Rust"]},
        {"name": "Bob", "role": "Designer"}
    ]
}

# JSON 크기를 제한하여 분할
splitter = RecursiveJsonSplitter(max_chunk_size=50)

# JSON 객체를 분할된 텍스트(또는 딕셔너리) 리스트로 반환
chunks = splitter.split_json(json_data)

for chunk in chunks:
    print(chunk)

# 출력 (구조를 유지하며 잘림):
# {'company': 'AI Corp'}
# {'employees': [{'name': 'Alice', 'role': 'Developer'}]}
# {'employees': [{'skills': ['Python', 'C++', 'Java', 'Go', 'Rust']}]}
# {'employees': [{'name': 'Bob', 'role': 'Designer'}]}





































################################################################################################
# Advanced Spliter Techniques
################################################################################################
# 1) 이중 분할 (Two-Pass Splitting)
#   실제 RAG 시스템을 구축할 때는 Markdown 분할기로 먼저 구조를 잡고, 
#   내용이 너무 길면 Recursive 분할기로 한 번 더 자르는 방식을 가장 많이 사용합니다.
# -------------------------------------------------------------------------------------------------------------------

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 불러온 마크다운 문서라고 가정 (내용이 꽤 길다고 가정)
markdown_document = """
# 1장. RAG 시스템 가이드

RAG(Retrieval-Augmented Generation)는 대규모 언어 모델의 환각 현상을 줄이고, 
최신 외부 데이터를 활용하여 정확한 답변을 생성하는 기술입니다. 
이 기술은 기업의 내부 문서를 기반으로 한 사내 챗봇 구축에 매우 유용하게 사용됩니다.

## 1.1 텍스트 분할의 중요성

텍스트 분할(Text Splitting)은 RAG 파이프라인에서 검색 품질을 결정짓는 핵심 단계입니다. 
문서를 너무 크게 자르면 검색의 정확도가 떨어지고 노이즈가 섞이게 됩니다. 
반대로 너무 작게 자르면 문맥이 끊어져서 LLM이 질문의 의도를 파악하지 못할 수 있습니다. 
따라서 적절한 청크 크기(Chunk Size)와 겹침(Overlap)을 설정하는 것이 중요합니다.
"""

# [1차 분할] 마크다운 헤더 기준으로 자르기
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# split_text()를 사용하여 텍스트를 Document 객체 리스트로 변환
md_header_splits = markdown_splitter.split_text(markdown_document)

print(f"1차 분할 후 청크 개수: {len(md_header_splits)}개\n")
# 결과: 2개 (1장 내용, 1.1 내용)


# [2차 분할] 글자 수 기준으로 한 번 더 자르기
# 테스트를 위해 청크 크기를 60글자로 아주 작게 설정
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=60, 
    chunk_overlap=15
)

# ⭐ 핵심: split_text가 아니라 split_documents를 사용해야 메타데이터가 유지됩니다!
final_splits = text_splitter.split_documents(md_header_splits)

print(f"2차 분할 후 최종 청크 개수: {len(final_splits)}개\n")

# 최종 결과 확인
for i, doc in enumerate(final_splits):
    print(f"--- 최종 청크 {i+1} ---")
    print(f"메타데이터: {doc.metadata}")
    print(f"내용: {doc.page_content}\n")




################################################################################################
# 2) 순차적 하이브리드 분할 (Sequential Splitting) : SemanticChunker → RecursiveCharacterTextSplitter
#   1차로 의미 기반 분할을 하고, 2차로 글자 수 기반 분할을 하는 방식입니다.
#
#   왜 필요한가? SemanticChunker는 의미가 이어지는 한 청크를 계속 길게 가져갑니다. 
#           만약 문서 내에 같은 주제를 다루는 내용이 3~4페이지에 걸쳐 이어진다면, 생성된 청크가 LLM의 입력 제한(Context Window)을 초과할 위험이 있습니다. 이를 방지하기 위한 **안전장치(Safety Net)**로 RecursiveCharacterTextSplitter를 사용합니다.
#
#   작동 방식:
#        SemanticChunker로 문서를 의미 단위(주제별)로 크게 덩어리 짓습니다.
#        RecursiveCharacterTextSplitter를 사용해, 1차에서 만들어진 덩어리 중 너무 큰 덩어리만 지정된 최대 크기로 강제로 자릅니다.
#
#   언제 사용?
#       비용과 속도가 중요할 때 : 벡터 DB를 하나만 유지해도 되므로 관리 포인트가 적고, 검색 속도가 빠릅니다.
# -------------------------------------------------------------------------------------------------------------------


from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

docs = [
    Document(
        page_content="""
# RAG 구축 가이드

RAG는 Retrieval-Augmented Generation의 약자이다.
이 방식은 대규모 언어 모델이 외부 지식을 검색한 뒤 그 결과를 바탕으로 답변을 생성하도록 만든다.
기본적인 생성형 AI는 학습 시점 이후의 최신 정보를 알지 못하거나, 내부 문서를 직접 참조하지 못하는 한계가 있다.
RAG는 이 문제를 해결하기 위해 문서를 검색하고, 검색된 문맥을 프롬프트에 삽입한다.

RAG의 핵심 구성요소는 문서 로더, 텍스트 분할기, 임베딩 모델, 벡터 저장소, 검색기, 그리고 생성 모델이다.
이 중 텍스트 분할은 검색 품질에 큰 영향을 준다.
문서를 너무 크게 나누면 여러 주제가 한 청크에 섞여 검색 정밀도가 낮아진다.
반대로 너무 작게 나누면 문맥이 잘려서 답변 생성 시 필요한 배경 정보가 사라질 수 있다.
적절한 chunk_size와 overlap을 정하는 것은 실험적으로 최적화해야 한다.

Chunk overlap은 앞뒤 청크가 일부 텍스트를 공유하도록 하는 설정이다.
예를 들어 overlap이 100이면 앞 청크 끝부분 100자 정도가 다음 청크 앞부분에도 들어간다.
이 방식은 문장이 청크 경계에서 끊어지는 문제를 줄인다.
하지만 overlap이 너무 크면 저장되는 총 토큰 수가 증가해 비용과 검색 노이즈가 커질 수 있다.
        """,
        metadata={"source": "rag_guide.md", "section": "intro"}
    ),
    Document(
        page_content="""
# Chunking 전략 상세

문자 기반 분할은 가장 단순한 방식이다.
대표적으로 RecursiveCharacterTextSplitter는 문단, 줄바꿈, 공백, 문자 순서로 재귀적으로 문서를 분할한다.
이 방식은 속도가 빠르고 예측 가능성이 높아서 대부분의 RAG 프로젝트에서 기본값으로 사용된다.
특히 구조화되지 않은 일반 텍스트 문서를 다룰 때 안정적인 성능을 보인다.

의미 기반 분할은 문장 임베딩을 활용하여 의미가 비슷한 문장끼리 묶는다.
SemanticChunker는 문장 간 임베딩 거리 변화를 계산하여 주제가 바뀌는 지점을 경계로 삼는다.
예를 들어 앞부분에서 텍스트 분할을 설명하다가 갑자기 벡터 데이터베이스의 인덱싱 전략을 설명하기 시작하면,
그 지점을 새로운 청크 시작점으로 인식할 가능성이 높다.

의미 기반 분할은 검색 품질을 높일 수 있지만 비용이 더 든다.
분할 자체를 위해 임베딩 계산이 필요하기 때문이다.
또한 문서가 짧거나 주제 전환이 적다면 큰 이점을 보지 못할 수도 있다.
따라서 실제 프로젝트에서는 문자 기반 분할을 먼저 사용해 baseline을 만들고,
검색 실패 사례가 반복될 때 의미 기반 분할을 추가 검토하는 접근이 현실적이다.
        """,
        metadata={"source": "rag_guide.md", "section": "chunking"}
    ),
    Document(
        page_content="""
# Retrieval 튜닝

벡터 검색은 사용자의 질문과 유사한 청크를 찾아오는 단계이다.
그러나 유사도 검색만으로 항상 최적 결과를 보장하지는 않는다.
예를 들어 사용자가 특정 키워드를 정확히 포함한 문서를 찾고 싶을 때는 BM25 같은 sparse retrieval이 더 강할 수 있다.
반면 질문이 길고 추상적일수록 dense retrieval의 장점이 커진다.

이 때문에 최근에는 hybrid retrieval이 많이 사용된다.
즉, sparse retriever와 dense retriever를 함께 사용한 뒤 결과를 fusion하는 방식이다.
RRF(Reciprocal Rank Fusion)는 서로 다른 검색기 결과를 합칠 때 널리 쓰이는 알고리즘이다.
이 방식은 각 검색기의 top-k 결과 순위를 기반으로 최종 점수를 계산한다.

검색기 튜닝 시에는 chunking 방식과 retrieval 방식을 함께 봐야 한다.
chunking이 너무 거칠면 retriever가 아무리 좋아도 필요한 문맥을 정확히 찾기 어렵다.
반대로 chunking이 아주 정교해도 retriever 설정이 나쁘면 관련 문서를 놓칠 수 있다.
결국 RAG 성능은 splitter, embedding, retriever, prompt가 함께 결정한다.
        """,
        metadata={"source": "rag_guide.md", "section": "retrieval"}
    ),
]



# 1차 분할: 의미(Semantic) 기반으로 주제별로 묶기
semantic_splitter = SemanticChunker(embeddings)

# docs는 Document 객체의 리스트라고 가정
semantic_chunks = semantic_splitter.split_documents(docs) 

print("=== Semantic 1차 분할 결과 ===")
for i, doc in enumerate(semantic_chunks, 1):
    print(f"--- 최종 청크 {i+1} ---")
    print(f"메타데이터: {doc.metadata}")
    print(f"내용: {doc.page_content}\n")


# 2차 분할: 너무 큰 의미 덩어리를 물리적 크기로 제한하기
# chunk_size를 넉넉하게 주어, 의미가 잘게 쪼개지는 것을 최소화함
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)

final_chunks = recursive_splitter.split_documents(semantic_chunks)


# 최종 결과 확인
print("=== Recursive 최종분할 결과 ===")
for i, doc in enumerate(final_chunks):
    print(f"--- 최종 청크 {i+1} ---")
    print(f"메타데이터: {doc.metadata}")
    print(f"내용: {doc.page_content}\n")






################################################################################################
# 병렬 앙상블 검색 (Parallel Ensemble Retrieval) 
#     두 스플리터의 결과를 억지로 하나의 기준으로 합치는 것이 아니라, 각각 독립적으로 벡터 DB에 저장한 뒤 검색 단계에서 두 결과를 융합하는 방식입니다.
#
#   왜 필요한가?
#       Recursive 방식은 특정 키워드나 짧은 문맥을 정확히 찾아내는 데 유리합니다.
#       Semantic 방식은 문서의 전반적인 주제나 포괄적인 의미를 찾아내는 데 유리합니다.
#       이 둘을 병렬로 검색하여 결과를 합치면(Ensemble), 단답형 질문과 포괄적 질문 모두에 강한 RAG 시스템이 완성됩니다.
#   
#   작동 방식:
#       원본 문서를 Recursive로 쪼개서 '벡터 DB A'에 넣습니다.
#       동일한 원본 문서를 Semantic으로 쪼개서 '벡터 DB B'에 넣습니다.
#       사용자가 질문하면 A와 B에서 각각 관련된 청크를 검색해 옵니다.
#       RRF(Reciprocal Rank Fusion) 같은 알고리즘을 사용해 양쪽에서 공통으로 중요하다고 판단한 청크의 순위를 높여 최종 컨텍스트를 구성합니다
# 
#   언제 사용?
#       검색 품질(정확도)이 최우선일 때 : 실제 엔터프라이즈급 RAG 시스템(예: 사내 규정 챗봇, 법률/의료 AI)에서는 방법 2를 많이 사용합니다.
#                                     다만, 문서를 두 번 임베딩해야 하므로 초기 구축 비용(API 비용)이 2배로 들고, 벡터 DB 용량도 2배로 차지한다는 단점을 고려해야 합니다.
# -------------------------------------------------------------------------------------------------------------------
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import EnsembleRetriever, SelfQueryRetriever
# from langchain_community.retrievers import BM25Retriever


docs = [
    Document(
        page_content="""
사과, 배, 바나나, 오렌지와 같은 과일은 비타민과 식이섬유가 풍부하다.
과일 섭취는 건강 유지에 도움이 되며, 특히 아침 식사 대용으로 많이 사용된다.
과일의 당분은 천연 당이지만 과다 섭취는 주의해야 한다.
과일 보관 시에는 품목별 적정 온도와 습도를 고려해야 한다.

벡터 데이터베이스는 임베딩 벡터를 저장하고 유사도 검색을 수행하는 시스템이다.
대표적으로 FAISS, Chroma, Pinecone, Weaviate 등이 사용된다.
벡터 검색에서는 cosine similarity, dot product, euclidean distance 등의 거리 함수가 활용된다.
대규모 서비스에서는 인덱스 최적화와 검색 지연 시간 관리가 매우 중요하다.

자동차 엔진오일은 주행 거리와 운행 조건에 따라 주기적으로 교체해야 한다.
타이어 공기압 점검과 브레이크 패드 상태 확인도 안전 운전에 필수적이다.
차량 정비 이력은 중고차 가치에도 영향을 준다.
정기 점검은 예상치 못한 고장을 예방하는 데 도움이 된다.
        """,
        metadata={"source": "mixed_topics.txt"}
    )
]



# --- 1. Recursive 방식 파이프라인 구축 ---
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs_recursive = recursive_splitter.split_documents(docs)
vectorstore_rec = FAISS.from_documents(docs_recursive, embeddings)
# 검색기(Retriever) 생성
retriever_rec = vectorstore_rec.as_retriever(search_kwargs={"k": 3})



# --- 2. Semantic 방식 파이프라인 구축 ---
semantic_splitter = SemanticChunker(embeddings)
docs_semantic = semantic_splitter.split_documents(docs)
vectorstore_sem = FAISS.from_documents(docs_semantic, embeddings)
# 검색기(Retriever) 생성
retriever_sem = vectorstore_sem.as_retriever(search_kwargs={"k": 3})

# --- 3. 앙상블 검색기(Ensemble Retriever) 결합 ---
# weights를 통해 어떤 검색기의 결과에 더 가중치를 둘지 결정 (여기서는 5:5)
ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever_rec, retriever_sem],
    weights=[0.5, 0.5]
)

# --- 4. 실제 검색 실행 ---
query = "텍스트 분할이 검색 성능에 왜 중요한가?"
ensemble_results = ensemble_retriever.invoke(query)

# 결과 확인: Recursive에서 찾은 결과와 Semantic에서 찾은 결과가 
# 지능적으로 융합(재정렬)되어 반환됩니다.
for i, doc in enumerate(ensemble_results):
    print(f"Rank {i+1}: {doc.page_content[:150]}...")



