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
# RAG Process
# 1. Document Load : 문서를 로드합니다.
# 2. Document Split : 문서를 청크 단위로 분할합니다.
# 3. Embedding : 분할된 청크를 벡터로 변환합니다.
# 4. VectorDB 생성 : 벡터화된 청크를 데이터베이스에 저장합니다.
# 5. Retriever 생성 : 데이터베이스에서 관련성이 높은 청크를 검색하는 검색기를 생성합니다.
# 6. Prompt 생성 : 검색된 청크를 활용하여 프롬프트를 생성합니다.
# 7. LLM과 체인 생성 : 프롬프트를 활용하여 LLM과 체인을 생성합니다.
# 8. 체인 실행 : 문서에 대한 질의를 입력하고, 답변을 출력합니다.
##########################################################################################

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS      # vectorstore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# 단계 1: 문서 로드(Load Documents)
loader = PyMuPDFLoader(f"{folder_path}/database/SPRI_AI_Brief_2023년12월호_F.pdf")
docs = loader.load()
print(f"문서의 페이지수: {len(docs)}")


# docs[10].__dict__   # 문서 객체의 속성 확인 (meta data 확인)
# print( docs[1].page_content[:500] )  # 첫 페이지의 앞 500자 출력



# 단계 2: 문서 분할(Split Documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(docs)
print(f"분할된 청크의수: {len(split_documents)}")


# split_documents[10].__dict__   # 분할된 문서 객체의 속성 확인 (meta data 확인)
# print(split_documents[10].page_content)   # 분할된 문서 객체의 속성 확인 (meta data 확인)


# 단계 4: DB 생성(Create DB) 및 저장
# 벡터스토어를 생성합니다.
vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)


for doc in vectorstore.similarity_search("구글"):
    print(doc.page_content)
    print("\n" + "#" * 100 + "\n")
    
    
# 단계 5: 검색기(Retriever) 생성
# 문서에 포함되어 있는 정보를 검색하고 생성합니다.
retriever = vectorstore.as_retriever()

# invoke
responses = retriever.invoke("삼성전자가 자체 개발한 AI 의 이름은?")        # 관련성이 높은 chunk들을 반환

for res in responses:
    print(res)
    # print(res.page_content)
    print("\n" + "#" * 100 + "\n")
    




# 단계 6: 프롬프트 생성(Create Prompt)
# 프롬프트를 생성합니다.
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Answer in Korean.

#Context: 
{context}

#Question:
{question}

#Answer:"""
)


# 단계 8: 체인(Chain) 생성
chain = ({
        "context": retriever,          # retriever에서 반환된 chunk들을 'context'라는 키로 프롬프트에 전달
        "question": RunnablePassthrough()      # 입력받은 질문을 그대로 'question'이라는 키로 프롬프트에 전달
        }
        | prompt
        | llm
        | StrOutputParser()
    )


# 체인 실행(Run Chain)
# 문서에 대한 질의를 입력하고, 답변을 출력합니다.
question = "삼성전자가 자체 개발한 AI 의 이름은?"

# invoke 출력
response = chain.invoke(question)
print(response)

# streaming 출력
response = StreamResponse(chain.stream(question))
print(response)

