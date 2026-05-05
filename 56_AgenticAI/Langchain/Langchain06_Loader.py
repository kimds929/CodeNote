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








##########################################################################################
##########################################################################################
# Document : LangChain 의 기본 문서 객체입니다.
#   - `page_content`: 문서의 내용을 나타내는 문열입니다.
#   - `metadata`: 문서의 메타데이터를 나타내는 딕셔너리입니다.


## Document Loader
# 다양한 파일의 형식으로부터 불러온 내용을 문서(Document) 객체로 변환하는 역할을 합니다.
#
### 주요 Loader 
#   - PyPDFLoader: PDF 파일을 로드하는 로더입니다.
#   - CSVLoader: CSV 파일을 로드하는 로더입니다.
#   - UnstructuredHTMLLoader: HTML 파일을 로드하는 로더입니다.
#   - JSONLoader: JSON 파일을 로드하는 로더입니다.
#   - TextLoader: 텍스트 파일을 로드하는 로더입니다.
#
### 통합형 Loader 인터페이스
# loader 객체 : 다양한 loader 도구를 활용하여 loader 객체를 생성한다.
# 생성된 loader 객체의 load() 함수는 전체 문서를 로드하여 반환한다.
# 생성된 loader 객체의 load_and_split() 함수는 splitter 도구를 활용하여 문서를 분할하여 반환한다.
# 반환된 document 객체는 page_content와 metadata 속성을 가진다.
#   page_content는 문서의 내용을 나타내는 문자열 이고, metadata는 문서의 메타정보를 담고있는 dictionary이다.



from langchain_core.documents import Document
document = Document(page_content="안녕하세요? 이건 랭체인의 도큐먼드 입니다")
# document.__dict__
# document.page_content
# dict(document)

# metadata에 속성 추가
document.metadata
document.metadata["source"] = "kimds929"
document.metadata["page"] = 1
document.metadata["author"] = "kimds929"


dict(document).keys()

##########################################################################################
# PDF Loader #############################################################################
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, PDFPlumberLoader

# 로더 설정
# loader = PyPDFLoader(f"{folder_path}/database/SPRI_AI_Brief_2023년12월호_F.pdf")
# loader = PyMuPDFLoader(f"{folder_path}/database/SPRI_AI_Brief_2023년12월호_F.pdf")
loader = PDFPlumberLoader(f"{folder_path}/database/SPRI_AI_Brief_2023년12월호_F.pdf")       # ★ PDFPlumberLoader는 다양한 metadata를 제공


# (STEP 1: 문서 로드(Load Documents))
# ---------------------------------------------------------------------------------------
### load()
# - 문서를 로드하여 반환합니다.
# - 반환된 결과는 `List[Document]` 형태입니다.

# PDF 로더
docs = loader.load()
len(docs)       # 로드된 문서의 수 확인

# 일부 문서 확인
docs[5].__dict__   # 문서 객체의 속성 확인 (meta data 확인)
print( docs[5].page_content[:500] )  # 페이지의 앞 500자 출력


# ---------------------------------------------------------------------------------------
### load_and_split()
# - splitter 를 사용하여 문서를 분할하고 반환합니다.
# - 반환된 결과는 `List[Document]` 형태입니다.

from langchain_text_splitters import RecursiveCharacterTextSplitter

# 문열 분할기 설정
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)

# 문서 분할
split_docs = loader.load_and_split(text_splitter=text_splitter)

print(f"문서의 길이: {len(split_docs)}")        # load후 분할된 문서의 수 확인

# 첫번째 문서 확인
split_docs[15].__dict__   # 분할된 문서 객체의 속성 확인 (meta data 확인)
print(split_docs[15].page_content[:500])   # 분할된 문서 객체의 속성 확인 (meta data 확인)


# ---------------------------------------------------------------------------------------
### lazy_load()
# - generator 방식으로 문서를 로드합니다.

loader.lazy_load()

# generator 방식으로 문서 로드
for doc in loader.lazy_load():
    print(doc.metadata)
    
    
# ---------------------------------------------------------------------------------------
### aload()
# - 비동기(Async) 방식의 문서 로드

# 문서를 async 방식으로 로드
adocs = loader.aload()

# 문서 로드
await adocs
# ---------------------------------------------------------------------------------------


messages = "삼성전자가 자체 개발한 AI 의 이름은?"


##########################################################################################
# Web Loader #############################################################################
import bs4
from langchain_community.document_loaders import WebBaseLoader


# - `bs4`는 웹 페이지를 파싱하기 위한 라이브러리입니다.
# - `langchain`은 AI와 관련된 다양한 기능을 제공하는 라이브러리로, 
#   여기서는 특히 텍스트 분할(`RecursiveCharacterTextSplitter`), 문서 로딩(`WebBaseLoader`), 벡터 저장(`Chroma`, `FAISS`), 출력 파싱(`StrOutputParser`), 실행 가능한 패스스루(`RunnablePassthrough`) 등을 다룹니다.
# - `langchain_openai` 모듈을 통해 OpenAI의 챗봇(`ChatOpenAI`)과 임베딩(`OpenAIEmbeddings`) 기능을 사용할 수 있습니다.



# (STEP 1: 문서 로드(Load Documents)) 뉴스기사 내용을 로드하고, 청크로 나누고, 인덱싱합니다.
loader = WebBaseLoader(
    web_paths=("https://n.news.naver.com/article/437/0000378416",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
    requests_kwargs={"verify": False}       ## SSL 검증을 비활성화합니다.
)



docs = loader.load()
print(f"문서의 수: {len(docs)}")
print(docs[0].metadata)
print(docs[0].page_content)
docs


messages = "부영그룹의 출산 장려 정책에 대해 설명해주세요."
messages = "부영그룹은 출산 직원에게 얼마의 지원을 제공하나요?"
messages = "정부의 저출생 대책을 bullet points 형식으로 작성해 주세요."
messages = "부영그룹의 임직원 숫자는 몇명인가요?"

# -------------------------------------------------------------------------------------------

loader = WebBaseLoader(
    web_paths=("http://www.newstomato.com/ReadNews.aspx?no=1299396&inflow=N",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["rn_stitle", "rns_text"]},
        )
    ),
    requests_kwargs={"verify": False}       ## SSL 검증을 비활성화합니다.
)
# ssl 인증 우회
# loader.requests_kwargs = {"verify": True}



docs = loader.load()
print(f"문서의 수: {len(docs)}")
print(docs[0].metadata)
print(docs[0].page_content)
docs


messages = "뉴스기사내용을 요약해주세요"
messages = "하청 노동자 직고용 인원수는?"
messages = "새로 직고용된 노동자는 어떤 직군으로 분류되나요?"
messages = "직고용 과정에서 어떤 문제가 있나요? 문제를 나열해주세요"






##########################################################################################
# DataFrame관련 Loader
#   - 의미론적 유사도 검색 (Semantic Search)
#   - 한계 : 정확한 수치 계산, 통계, 정밀한 조건 검색 불가
#       → 계산 및 조건 검색은 Pandas Agent 혹은 SQL Agent이용 필요
#
# 1. 텍스트가 중심인 데이터 (Text-Heavy Data)
#   - DataFrame내에 길고 서술적인 텍스트(자연어)로 채워진 경우.
#     ex. 고객 리뷰 데이터, 상담이력, 제품상세설명서
#
# 2. FAQ (자주 묻는 질문) 및 사내 지식베이스
#   - DataFrame내 질문-답변 형식으로 되어있는 경우
#     ex. 챗봇 구축
#
# 3. 메타데이터 필터링이 필요한 문서 검색 (DataFrameLoader의 진가)
#   - DataFrameLoader는 단순히 데이터프레임을 로드하는 것을 넘어, 특정 컬럼은 '본문(Content)'으로 임베딩하고, 나머지 컬럼은 '메타데이터(Metadata)'로 분리하는 강력한 기능을 제공합니
#     ex. 수만 건의 뉴스 기사 CSV (컬럼: 기사_본문, 작성일, 기자명, 카테고리)
#

# csv Loader #############################################################################

from langchain_community.document_loaders.csv_loader import CSVLoader

# CSV 로더 생성
loader = CSVLoader(file_path=f"{folder_path}/database/titanic_original.csv",
        csv_args={
        "delimiter": ",",  # 구분자
        })

# 데이터 로드
docs = loader.load()

print(len(docs))
print(docs[0].metadata)
print(docs[1].page_content)


# 문서 전체를 XML 문서 형식으로 처리하려는 경우
# - 참고: 0번째 문서는 헤더 정보이기 때문에 스킵합니다.

# 1줄 XML parsing 예시
row = docs[1].page_content.split("\n")
row_str = "<row>"
for element in row:
    splitted_element = element.split(":")
    value = splitted_element[-1]
    col = ":".join(splitted_element[:-1])
    row_str += f"<{col}>{value.strip()}</{col}>"
row_str += "</row>"
print(row_str)



# 전체 XML parsing
for doc in docs[1:]:
    row = doc.page_content.split("\n")
    row_str = "<row>"
    for element in row:
        splitted_element = element.split(":")
        value = splitted_element[-1]
        col = ":".join(splitted_element[:-1])
        row_str += f"<{col}>{value.strip()}</{col}>"
    row_str += "</row>"
    print(row_str)
    # 변환된 XML 문자열을 Document의 본문으로 덮어씌움
    doc.page_content = row_str


messages = "20대 남자중에 살아남은 사람들을 알려줘."



# Pandas DataFrame Loader #############################################################################
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
df = pd.read_csv(f"{folder_path}/database/titanic_original.csv", sep=',')
df.head(5)
df['name'] = df['name'].fillna("Unknown")


# 데이터 프레임 로더 설정, 페이지 내용 컬럼 지정
loader = DataFrameLoader(df, page_content_column='name')

# 문서 로드
docs = loader.load()

# 데이터 출력
print(docs[0].page_content)

# 메타데이터 출력
print(docs[0].metadata)

messages = "20대 남자중에 살아남은 사람들을 알려줘."


##########################################################################################
# Arxiv-Loader #############################################################################

from langchain_community.document_loaders import ArxivLoader


# Query 에 검색하고자 하는 논문의 주제를 입력합니다.
loader = ArxivLoader(
    query="Chain of thought",
    load_max_docs=2,  # 최대 문서 수
    load_all_available_meta=True,  # 메타데이터 전체 로드 여부
)

# 문서 로드 결과출력
docs = loader.load()
docs

# 문서 메타데이터 출력
docs[0].metadata

# ------------------------------------------------------------------------------------------------
# `load_all_available_meta=False` 인 경우 메타데이터는 전체가 아닌 일부만 출력됩니다.
# Query 에 검색하고자 하는 논문의 주제를 입력합니다.
loader = ArxivLoader(
    query="ChatGPT",
    load_max_docs=2,  # 최대 문서 수
    load_all_available_meta=False,  # 메타데이터 전체 로드 여부
)

# 문서 로드 결과출력
docs = loader.load()

# 문서 메타데이터 출력
docs[0].metadata

# ------------------------------------------------------------------------------------------------
## 요약(summary)
# - 논문의 전체 내용이 아닌 요약본을 출력하고자 한다면, `get_summaries_as_docs()` 함수를 호출하면 됩니다.
# 문서 요약 로딩
docs = loader.get_summaries_as_docs()

# 첫 번째 문서 접근
print(docs[0].page_content)

##########################################################################################
# Directory-Loader #############################################################################






##########################################################################################
##########################################################################################
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# (STEP 2: 문서 분할(Split Documents))
# ---------------------------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"분할된 청크의수: {len(split_documents)}")


# (STEP 3: DB 생성(Create DB) 및 저장)
# ---------------------------------------------------------------------------------------
# 벡터스토어를 생성
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)

# # Similarity Search
# for doc in vectorstore.similarity_search("구글"):
#     print(doc.page_content)
#     print("\n" + "#" * 100 + "\n")


# (STEP 4 : 검색기(Retriever) 생성)
# ---------------------------------------------------------------------------------------
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# invoke
responses = retriever.invoke(messages)        # 관련성이 높은 chunk들을 반환

# for res in responses:
#     print(res)
#     # print(res.page_content)
#     print("\n" + "#" * 100 + "\n")


# (STEP 5: 프롬프트 생성(Create Prompt))
# ---------------------------------------------------------------------------------------
from langchain_core.prompts import PromptTemplate

# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 
당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.


#Context: 
{context}

#Question:
{question}

#Answer:"""
)

# (STEP 6: 체인(Chain) 생성)
# ---------------------------------------------------------------------------------------
chain = ({
        "context": retriever,          # retriever에서 반환된 chunk들을 'context'라는 키로 프롬프트에 전달
        "question": RunnablePassthrough()      # 입력받은 질문을 그대로 'question'이라는 키로 프롬프트에 전달
        }
        | prompt
        | llm
        | StrOutputParser()
    )


# (STEP 7: 체인 실행(Run Chain))
# ---------------------------------------------------------------------------------------
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.

# # invoke 출력
# response = chain.invoke(messages)
# print(response)

# streaming 출력
response = StreamResponse(chain.stream(messages))
print(response)

##########################################################################################
##########################################################################################