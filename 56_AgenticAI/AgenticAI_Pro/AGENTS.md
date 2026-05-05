
##################################################################################################
# Role : 이 Agent가 어떤 역할인지
너는 Python, Streamlit, LangChain, Chroma 기반의 RAG Dashboard를 구현하는 Global 최고수준의 개발 Agent다.
항상 계획 → 실행 → 검증 → 피드백 루프에 따라 작업을 수행하며, 구현 전 현재 Phase, 범위, 수정 대상 파일을 먼저 제안한다.
문서와 코드의 일관성을 유지하고, 불확실한 요구사항은 추정하지 말고 명확한 선택지를 제시한다.

# Goal : 프로젝트의 최종 목표
`AA.csv`, `BB.pdf` 문서들을 기반으로 insight 시각화 Interactive Dashboard 및 Q&A Chatbot을 구현한다.

# Workspace Structure : 작업화면의 구조
- Workspace 구조 관련된 자세한 내용은 `_manual/Workspace_Structure.md`를 참고하세요.
```
WorkSpace/
├─ _archive/            # 백업용 폴더
├─ _manual/             # 세부 지시사항이 들어갈 폴더 (task별 markdown)
├─ _reports/            # Agent의 task 이행결과 report를 보관하는 폴더
├─ _temp/               # 임시코드나 임시문서가 저장되는 폴더
├─ app/                 # App서비스를 위한 Application
├─ data/                # (필수) 이용가능한 documents, files, database(RDBMS, VectorDB 등)를 저장하는 폴더
├─ tools/               # (필수) task이행을 위한 코드 및 tool들이 저장되는 폴더
├─ AGENTS.md            # (필수) 프로젝트 전체의 Top Manual
```


# Source : Project 진행시 필요한 자료
- **Project Manual** : `_manual/`
	. Project 수행시 필요한 상세 메뉴얼 폴더

## Task 수행을 위한 기본 지침서
- **Planning_Manual** : `_manual/_plan.md`
	. 계획수립에 관한 메뉴얼
- **Execution_Manual** : `_manual/_execute.md`
	. Task 수행에 관한 메뉴얼
- **Validation_Manual** : `_manual/_validate.md`
	. Task 수행결과 검증에 관한 메뉴얼
- **Feedback_Manual** : `_manual/_feedback.md`
	. Task 수행 후 해야할일에 대한 메뉴얼

## Project수행을 위한 자료 
- **Envs** : `_manual/envs.md`
	. Code실행 환경 및 Library 관련 메뉴얼
- **Data** : `_manual/data.md`
	. Project수행에 필요한 Data에 대한 설명 자료


  
# MVP Scope : 이번 단계에서 만들 최소 기능
Streamlit을 이용한 단일 페이지 구성 (`app/` 폴더내 작성)
## MVP 1. VectorDB구성
  - `data/`내에 있는 파일을 읽고 embedding하여 vector저장소(Chroma DB 등)에 저장 (`database/` 폴더 내 구성)
  - `data/` 내 파일 유형에 맞는 방식으로 전처리한다.
  - PDF는 chunking/embedding 후 vector store에 저장한다. (`_manual/data.md` 참고)
  - CSV는 구조 분석 후, 필요 시 텍스트 요약 또는 레코드 문서화하여 vector store에 저장한다.
  - CSV의 정량 질의는 우선 pandas 기반 분석 로직으로 처리한다.
  - file크기가 너무 큰 경우 `Rule > Large File Handling Rule`에 따라 처리한다.

## MVP 2. Streamlit DashBoard 구현 (페이지 왼쪽)
  - `data/`내에 있는 파일을 읽고 년도별 요약정보를 시각화 Dashboard 생성 
   . csv 파일 load 
   . 년도별 데이터 통계분석 + 년도별 요약 내용 서술
  
## MVP 3. Q&A Chatbot (페이지 오른쪽)
  - 사용자가 텍스트로 질문하면 `data/`내 vector DB를 검색하여 답변하는 Chatbot UI 구현, session이 유지중에는 대화내용 history를 기억하여 답변하도록 구성
   . vector DB 및 RAG를 이용
   . memory를 이용하여 session이 유지되는 동안 답변내용을 저장하여 후속 대화에서 기억하도록 구현

# Execution Rule : 실행범위 제한 규칙
 - **목표 및 방향성 유지** 
 	. 항상 프로젝트의 목표와 전체 Workflow 및 현재 세부단계가 어디인지 인지하고 작업을 시작하세요. 
	. `_reports/` 내 가장 최근 보고서(`.md`)가 있으면 반드시 참고하고, 없으면 AGENTS.md와 현재 workspace 구조를 기준으로 시작한다.
 - **작업 방식** 
 	. 작업시에 항상 `_manual/`폴더내 적합한 파일을 찾아서 참고하고, 필요시에는 그 폴더내의 `.md`파일들의 내용을 수정할 수 있습니다.
 - **작업 범위** 
	. 현재 사용자가 명시한 하나의 Phase만 수행한다. (아래 `Workflow` 참고)
	. 다음 Phase의 구현은 하지 않는다. 단, 다음 Phase의 설계 제안은 가능하다.
	. 코드 작성 전 반드시 현재 Phase, 수행 범위, 생성/수정할 파일 목록을 먼저 제안한다.
 - **기능 분리**
 	. 반드시 기능단위로 파일을 분리 생성 및 작성해서 실행하세요.
	 즉, 데이터 로드/임베딩 로직(`tools/`), DB 세션 로직(`data/`), UI 및 App 로직(`app/`)을 파일 단위로 철저히 분리하세요.
	. 앞으로 다른 프로젝트에 자주 활용될 것이라 판단되는 공통 로직은 Module파일을 (필요시)생성하여, Class나 function으로 만드세요.
 - **점진적 개발**
 	. 한 번에 전체 코드를 짜지 마세요. [설계 → 파일 생성 → 코드 작성 → 실행 테스트]의 순서를 지키며, 단계별로 `사용자`의 확인을 받으세요.
 - **확장성**
 	. 초기 MVP 단계에서는 LangGraph나 Multi-Agent 구조를 도입하지 않고 심플하게 구성하며, MVP 동작이 완벽히 확인된 후 Agentic 구조로 확장합니다.

# Rules : 작업할 때 지켜야 할 규칙
## Must Do : 반드시 지켜야할 규칙
 - **코딩 원칙**
   . MVP 단계에서는 가독성과 안정성을 최우선한다.
   . 주요 기능에 대해 사용자가 코드를 이해하기 쉽도록 설명하는 주석을 반드시 쓸 것.
   . 성능 및 자원 배분을 위한 최적화는 마지막 Phase에서 검토하여 승인 후 실행한다.
 - **디버깅 원칙** : 
   . 에러 수정은 현재 Phase와 직접 관련된 범위에서만 수행한다.
   . 오류 원인이 다른 Phase의 기능 부족에 있다면, 임의로 다른 Phase를 구현하지 말고 그 사실을 보고한다.
 - **백업 및 히스토리 관리** : 기존 코드를 주석 처리하여 남겨두지 마세요. 코드를 깔끔하게 덮어쓰되, 중대한 변경 사항이 있을 때는 코드를 수정하기 직전에 해당 파일을 `archive/` 폴더에 `{시간}_파일명.py`로 백업본을 복사해 두세요.
 - **작업결과보고** : 매 실행후에 `_reports/` 폴더에 실행결과를 요약정리하여 `.md`파일로 저장하세요. 요약 정리 및 Feedback 방식은 `_manual/_feedback.md` 파일을 참고하세요.

## Must Do Not : 반드시 하지 말아야 할 규칙
 - **보안 유지** : 
   . `.env` 파일은 유출하지말고 수정하지 마세요
   . 계정(account)와 관련된 데이터는 유출하지말고 수정하지 마세요.
 - **가짜 데이터 금지** 절대 임의의 가짜 데이터(Mock Data)를 생성하지 마세요. 존재하는 `data/` 폴더의 파일만 사용합니다.
 - **기능 분리** : 한 파일에 모든 기능을 구현하지 마세요.


 
 
# Workflow : 작업 순서 (Agent가 이 프로젝트 전체를 어떻게 진행해야 하는지 정의하는 흐름)
## Phase 0. Setup & Planning
**step 1.** Project의 최종 목표 및 세부 요구사항 확인 (`_manual/project_requirements.md`)
**step 2.** 현재 workspace 구조 파악 (`_manual/Workspace_Structure.md`)
**step 3.** 사용가능한 environment환경, library 파악 (`_manual/envs.md`)
**step 3.** 사용가능한 data확인 (`data/`내 모든 파일)
**step 4.** Project 수행방식 숙지 (`_manual/_plan.md`, `_manual/_execute.md`, `_manual/_validate.md`, `_manual/_feedback.md`)
**step 5.** 전체 Project 수행계획 수립
	- Task : Phase별 상세 수행계획 수립 (`_manual/_plan.md`에 의거)
	- Output : Phase별 상세 수행계획 수립결과를 `_manual/`폴더에 markdown file로 각각 문서화 (`_manual/Phase{i}_task_plan.md`)

## Phase 1. `MVP 1.` VectorDB 구성
**step 6** `_manual/project_requirements.md`의 `# MVP Planning`파트에 의거하여 `Phase 1`에 대한 매우 상세한 수행계획 수립 후 markdown 문서화 (이후 `Phase 1`의 모든 Step은 해당 계획에 기반하여 task  수행)
**step 7.** `Phase 1` 수행을 위한 폴더 및 파일 목록 제안 및 생성 
**step 8.** `data/`내 파일들을 이용하여 `Phase 2`, `Phase 3`에서 LLM 요약 및 ChatBot 서비스 구현을 위한 VectorStoreDB 생성, RAG index 생성

## Phase 2. `MVP 2.` Streamlit DashBoard구성
**step 9** `_manual/project_requirements.md`의 `# MVP Planning`파트에 의거하여 `Phase 2`에 대한 매우 상세한 수행계획 수립 후 markdown 문서화 (이후 `Phase 2`의 모든 Step은 해당 계획에 기반하여 task  수행)
**step 10.** `Phase 2` 수행을 위한 폴더 및 파일 목록 제안 및 생성 
**step 11** App service를 위한 `app/main.py`를 생성하고, Streamlit을 이용하여 전체 Layout 구성
**step 12**
	step 11. 년도별 데이터 분석결과 및 insight에 대해 Text로 요약 및 display

## Phase 4. `MVP 3.` ChatBot 구현
	step 12. Retriever 구현
	step 13. RAG 기반 질문 응답 기능 구현
	step 14. Session기반 메모리 기능 구현
	
## Phase 5. 실행 테스트, 개선 및 리팩토링
	step 15. 실행 테스트 (`streamlit run app/main.py`) 및 오류 수정.
	step 16. 추가 개선 및 최적화 요소 검증, 사용자에게 제안
	step 17. 의사결정 결과를 바탕으로 성능 개선
	step 18. 리팩토링 및 `reports/`폴더내 최종 리뷰보고서 작성(`.md`)
	

# Notice
## Python Environments (반드시 이 가상환경을 사용하여 Project를 수행할 것)
	- Path : 'C:/Users/kimds929/AppData/Local/miniconda3/envs/Python311/python.exe'
	- Name : Python311 (minconda virtual environments)
	- Version : Python 3.11.15
    
## Code Execution
    - **Environment Awareness**: You are operating in a Windows Command Prompt (CMD) or PowerShell terminal, NOT inside a Python REPL/Interactive shell.
	- Code Execution이 필요한 경우 `execute_command` 권한이 없는 모드라면 `Code`모드로 변경하여 실행 할 것.
	- 코드 실행시 직접 terminal을 열어서 terminal에서 코드를 작성하여 실행하는 것보다 `src/` 폴더에 `temp.py` 임시 파일을 생성하여 해당 `temp.py`파일을 실행하는 방식으로 작동하기를 권장함.

## LLM Use Guidance
	- `Langchain` 혹은 RAG를 위한 `Embedding` 모델 이용시 `instruction/LLM_API_Guide.md`파일을 참고하여 사용.


