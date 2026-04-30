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


dict(docs[0]).keys()
# ---------------------------------------------------------------------------------------
### load()
# - 문서를 로드하여 반환합니다.
# - 반환된 결과는 `List[Document]` 형태입니다.

from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader, PDFPlumberLoader

# 로더 설정
# loader = PyPDFLoader(f"{folder_path}/database/SPRI_AI_Brief_2023년12월호_F.pdf")
# loader = PyMuPDFLoader(f"{folder_path}/database/SPRI_AI_Brief_2023년12월호_F.pdf")
loader = PDFPlumberLoader(f"{folder_path}/database/SPRI_AI_Brief_2023년12월호_F.pdf")       # ★ PDFPlumberLoader는 다양한 metadata를 제공

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

