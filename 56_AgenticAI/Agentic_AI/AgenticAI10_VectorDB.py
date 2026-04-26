









#######################################################################################################################

#  RAG (검색 증강 생성)에서의 답변 생성기 역할 사내 문서를 검색해서 답변을 생성하는 RAG 시스템을 구축

# # 예시: retriever는 이미 벡터 DB와 연결되어 있다고 가정
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain

# # 문서들을 조합해서 프롬프트에 넣는 체인 (llm 자리에 PoscoGPT 사용)
# combine_docs_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# rag_chain.invoke({"input": "포스코의 2050 탄소중립 목표는?"})




from langchain_community.vectorstores import FAISS

embeddings = PgptEmbeddings(
    api_key=os.getenv("API_KEY"),
    emp_no=os.getenv("EMP_NO"),
    model_name='text-embedding-ada-002'
)

# 테스트: 텍스트를 벡터(숫자 배열)로 잘 변환하는지 확인
vector = embeddings.embed_query("포스코의 핵심 가치는?")
print(f"✅ 임베딩 성공! 벡터 길이: {len(vector)}")
print(f"✅ 벡터 데이터 일부: {vector[:5]} ...")


# 2. 사내 규정이나 매뉴얼 데이터 (실제로는 PDF, Word 등에서 추출한 텍스트)
company_docs = [
    "회사 출근 시간은 오전 9시부터 10시 사이 자율 출근입니다.",
    "연차 신청은 최소 3일 전에 그룹웨어를 통해 결재를 받아야 합니다.",
    "식대 지원은 일 15,000원이며, 법인 카드로 결제해야 합니다."
]

# 3. 텍스트를 임베딩하여 FAISS 벡터 데이터베이스에 저장
# 이 과정에서 company_docs의 텍스트들이 PoscoEmbeddings를 거쳐 벡터로 변환된 후 FAISS에 저장됩니다.
print("벡터 DB를 생성 중입니다...")
vectorstore = FAISS.from_texts(company_docs, embeddings)
print("벡터 DB 생성 완료!\n")

# 4. 사용자 질문을 기반으로 유사도 검색 수행
user_question = "휴가 쓰려면 어떻게 해야 해?"

# 내부적으로 user_question을 embed_query로 벡터화한 뒤, DB에서 가장 가까운 벡터를 찾습니다.
docs = vectorstore.similarity_search(user_question, k=1) 

print(f"사용자 질문: {user_question}")
print(f"검색된 문서: {docs[0].page_content}")
# 출력 예상: "연차 신청은 최소 3일 전에 그룹웨어를 통해 결재를 받아야 합니다."
#######################################################################################################################











