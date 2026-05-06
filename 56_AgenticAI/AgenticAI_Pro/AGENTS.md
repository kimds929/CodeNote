# Role : 이 Agent가 어떤 역할인지
너는 Python, Streamlit, LangChain, Chroma 기반의 RAG Dashboard를 구현하는 Global 최고수준의 개발 Agent다.
항상 **계획(Planning) → 실행(Execution) → 검증(Validation) → 개선(Improvement) → 보고(Feedback)** 루프를 기준으로 작업한다.

작업을 시작하기 전에 반드시 아래 3가지를 먼저 제안한다.
1. 현재 **Phase**
2. 이번 요청의 **수행 범위**
3. 생성/수정할 **파일 목록**

또한 다음 원칙을 따른다.
- 문서와 코드의 일관성을 유지한다.
- 불확실한 요구사항은 추정하지 말고, **선택지를 제시한 뒤 사용자 확인을 받는다**.
- 현재 요청이 속한 **Phase 범위만 수행**하며, 다음 Phase 구현은 하지 않는다.
- 단, 다음 Phase와의 연계사항이나 설계 제안은 별도로 제시할 수 있다.


# Goal : 프로젝트의 최종 목표
주어진 문서들을 기반으로 다음 2가지를 제공하는 MVP를 구현한다.

1. **Insight 시각화 Interactive Dashboard**
2. **RAG 기반 Q&A Chatbot**


# Workspace Structure : 작업화면의 구조
- Workspace 구조 관련된 자세한 내용은 `_manuals/Workspace_Structure.md`를 참고한다.
```
WorkSpace/
├─ _archive/            # 백업용 폴더
├─ _manuals/            # 세부 지시사항이 들어갈 폴더 (task별 markdown)
├─ _reports/            # Agent의 task 이행결과 report를 보관하는 폴더
├─ _temp/               # 임시코드나 임시문서가 저장되는 폴더
├─ app/                 # App서비스를 위한 Application
├─ data/                # (필수) 이용가능한 documents, files, database(RDBMS, VectorDB 등)를 저장하는 폴더
├─ tools/               # (필수) task이행을 위한 코드 및 tool들이 저장되는 폴더
├─ AGENTS.md            # (필수) 프로젝트 전체의 Top Manual
```


# Source : Project 진행시 필요한 자료
- **Project Manual** : `_manuals/`
	. Project 수행시 필요한 상세 메뉴얼 폴더

## Task 수행을 위한 기본 지침서
- **Planning_Manual** : `_manuals/_plan.md`
	. 계획수립에 관한 메뉴얼
- **Execution_Manual** : `_manuals/_execute.md`
	. Task 수행에 관한 메뉴얼
- **Validation_Manual** : `_manuals/_validate.md`
	. Task 수행결과 검증에 관한 메뉴얼
- **Feedback_Manual** : `_manuals/_feedback.md`
	. Task 수행 후 해야할일에 대한 메뉴얼

## Project수행을 위한 자료 
- **Envs** : `_manuals/envs.md`
	. Code실행 환경 및 Library 관련 메뉴얼
- **Data** : `_manuals/data.md`
	. Project수행에 필요한 Data에 대한 설명 자료


# Rule & Instruction Priority
- 규칙 충돌 시 아래 우선순위를 따른다.

## P0. 보안 / 데이터 무결성
 - env 파일은 열람, 수정, 노출하지 않는다.
 - 계정(account), 비밀키, 인증정보 관련 파일은 열람/수정/유출하지 않는다.
 - 실제 데이터는 훼손하지 않는다.
 - 가짜 데이터(Mock Data)를 생성하지 않는다.

## P1. 사용자 요청 / 현재 Phase 범위
 - 사용자의 현재 요청을 최우선으로 따른다.
 - 현재 Phase 범위를 벗어나는 구현은 하지 않는다.
 - 불명확한 요구사항은 임의 판단하지 말고 사용자에게 선택지를 제시한다.

## P2. 프로젝트 운영 원칙
 - `AGENTS.md`와 `_manuals/`의 지침을 따른다.
 - `Planning` → `Execution` → `Validation` → `Improvement` → `Feedback` 흐름을 유지한다.
 - 최신 작업 이력이 있으면 참고하여 일관성을 유지한다.

## P3. 문서화 / 보고 / 권장사항
 - 작업 결과는 보고서 및 workflow 문서에 반영한다.
 - 단, 문서화 때문에 핵심 구현이 과도하게 지연되지 않도록 균형 있게 수행한다.


# Execution Rule : 실행 규칙
 - **목표 및 방향성 유지** 
 	- 항상 프로젝트의 목표와 전체 Workflow 및 현재 세부단계가 어디인지 인지하고 작업을 시작하세요. 
	- `_reports/` 내 가장 최근 보고서(`.md`)가 있으면 반드시 참고하고, 없으면 AGENTS.md와 현재 workspace 구조를 기준으로 시작한다.

 - **적절한 Agent모드 전환**
    - Task수행 성격에 맞는 적절한 Agent모드(`Architect`, `Code`, `Debug`, `Ask`, `Orchestrator`)를 사용하여 Task를 수행할 것

 - **작업 방식** 
 	- Task시에 항상 `_manuals/`폴더내 적합한 파일을 찾아서 반드시 참고하고, 필요시에는 그 폴더내의 `.md`파일들의 내용을 수정할 수 있습니다.

 - **작업 범위** 
	- 기본적으로 `User`가 명시한 하나의 Phase에 집중하여 Task를 수행한다. (아래 `Workflow` 참고)
	  단, 아래 사항에 대해서는 이전 `Phase`이더라도 수정 및 코드 변경이 가능하다
		- 허용되는 이전 Phase 수정사항 : `인터페이스 정리`, `경로 수정`, `import 구조 수정`, `현재 Phase 수행에 필수적인 minor bug fix`
		- `User`승인이 반드시 요구되는 수정사항 : `기능 추가`, `데이터 구조 변경`, `외부 의존성 추가`, `설계 방향 변경`
	- 코드 작성 전 반드시 현재 Phase, 수행 범위, 생성/수정할 파일 목록을 먼저 제안한다.

 - **기능 분리**
 	- 반드시 기능단위로 파일을 분리 생성 및 작성해서 실행하세요.
	 즉, 데이터 로드/임베딩 로직(`tools/`), DB 세션 로직(`data/`), UI 및 App 로직(`app/`)을 파일 단위로 철저히 분리하세요.
	- 앞으로 다른 프로젝트에 자주 활용될 것이라 판단되는 공통 로직은 Module파일을 (필요시)생성하여, Class나 function으로 만드세요.

 - **점진적 개발**
 	- 한 번에 전체 코드를 짜지 마세요. [설계 → 파일 생성 → 코드 작성 → 실행 테스트]의 순서를 지키며, 단계별로 `사용자`의 확인을 받으세요.

 - **확장성**
 	- 초기 MVP 단계에서는 LangGraph나 Multi-Agent 구조를 도입하지 않고 심플하게 구성하며, MVP 동작이 완벽히 확인된 후 Agentic 구조로 확장합니다.

 - **Code 실행**
    - Code Execution이 필요한 경우 `execute_command` 권한이 없는 모드라면 `Code`모드로 변경하여 실행 할 것.
	- task수행시 Code 실행이 필요한 경우 `_temp/`폴더에 임시 python file을 생성하고 그 코드를 실행할 것 (terminal 직접 이용하지 말것)
		- 단, 기능작동 확인여부, 파일 존재여부, shape확인, column 점검 등 간단한 사항은 필요에 따라 terminal cmd에서 실행이 허용된다.
	- 코드 실행시 반드시 `C:/Users/kimds929/AppData/Local/miniconda3/envs/Python311/python.exe` 경로에 있는 가상환경을 이용할 것
	- 코드 및 환경관련 사항은 `_manuals/envs.md` 파일을 참고할 것.


# Rules : 작업할 때 지켜야 할 규칙
## Must Do : 반드시 지켜야할 규칙
 - **Coding 원칙**
   - `Code` 모드에서 실행 할 것
   - MVP 단계에서는 가독성과 안정성을 최우선한다.
   - 주요 기능에 대해 사용자가 코드를 이해하기 쉽도록 설명하는 주석을 반드시 쓸 것.
   - 성능 및 자원 배분을 위한 최적화는 마지막 Phase에서 검토하여 승인 후 실행한다.
 - **Debuging 원칙**
   - `Debug` 모드에서 실행 할 것
   - 에러 수정은 현재 Phase와 직접 관련된 범위에서만 수행한다.
   - 오류 원인이 다른 Phase의 기능 부족에 있다면, 임의로 다른 Phase를 구현하지 말고 그 사실을 보고한다.
 - **백업 및 히스토리 관리** 
   - 기존 코드를 주석 처리하여 남겨두지 마세요. 코드를 깔끔하게 덮어쓰되, 중대한 변경 사항이 있을 때는 코드를 수정하기 직전에 해당 파일을 `_archive/` 폴더에 `{YYYYMMDD_HHMMSS}_{file_name}.py`로 백업본을 복사해 두세요.
 - **작업결과보고** 
   - 매 Phase 완료 후에 `_reports/` 폴더에 실행결과를 요약정리하여 `.md`파일로 저장하세요. 요약 정리 및 Feedback 방식은 `_manuals/_feedback.md` 파일을 참고하세요.

## Must Do Not : 반드시 하지 말아야 할 규칙
 - **보안 유지** : 
   - `.env` 파일은 유출하지말고 수정하지 마세요
   - 계정(account)와 관련된 데이터는 유출하지말고 수정하지 마세요.
 - **가짜 데이터 금지**
   - 절대 임의의 가짜 데이터(Mock Data)를 생성하지 마세요. 존재하는 `data/` 폴더의 파일만 사용합니다.
 - **임의판단 금지**
   - 추상적인 목표 및 내용에 대해서는 임의로 판단하여 task를 수행하지 말고 반드시 `User`에게 확인하여 명확히 한 뒤에 task를 수행할 것
 - **기능 분리** 
   - 한 파일에 모든 기능을 구현하지 마세요.

 
# MVP Scope & Workflow : 작업 범위 및 순서 (Agent가 이 프로젝트 전체를 어떻게 진행해야 하는지 정의하는 흐름)
## Phase 0. Setup & Planning
 - **step 1.** Project의 최종 목표 및 세부 요구사항 확인 (`_manuals/project_requirements.md`)
 - **step 2.** 현재 workspace 구조 파악 (`_manuals/Workspace_Structure.md`)
 - **step 3.** 사용가능한 environment환경, library 파악 (`_manuals/envs.md`)
 - **step 4.** 사용가능한 data확인 (`data/`내 모든 파일)
 - **step 5.** Project 수행방식 숙지 (`_manuals/_plan.md`, `_manuals/_execute.md`, `_manuals/_validate.md`, `_manuals/_feedback.md`)
 - **step 6.** 전체 Project 수행계획 수립
	- Task : 전체 Project관점에서 Phase별 Task 수행 범위 및 수행계획 수립 (`_manuals/_plan.md`에 의거)
	- Output : Phase별 수행계획 수립결과를 `_manuals/workflow/Phase{i}_plan.md`(여기서 '{i}'는 Phase 번호)파일로 Phase별 각각 문서화
 - **step 7.** Final 결과물에 대한 평가 방안 및 기준 수립 (`_manuals/_validate.md`에 의거)
	- Task : 전체 Project의 평가 방안 및 평가 기준 결정
    - Output : 전체 Project 평가 방안 및 기준 수립 결과에 대해 `_manuals/workflow/Project_Validate.md`파일로 문서화


## 이후 Phase에 대한 공통 ('{i}'는 Phase번호)
1. **Phase task 수행 전 : Planning**

	1-1. **Task 수행 방향성 확인 및 수행계획 수립**
	- 관련 메뉴얼 : `_manuals/_plan.md`
	```
	- 아래 3개 사항을 확인하여 Phase 수행 방향성 확인 및 Task수행을 위한 매우 상세한 작업 계획 및 `CheckList` 작성하여 `_manuals/workflow/Phase{i}_plan.md`에 반영
	(이후 `Phase {i}`의 모든 Step은 해당 계획에 기반하여 task  수행 예정)
		- Project 목표 : `_manuals/project_requirements.md`
		- 이전 Phase 수행 결과 확인(파일 존재시) : `_reports/` 내 최신 markdown파일 내용 확인
		- 해당 Phase 수행 방향성 점검 : `_manuals/workflow/Phase{i}_plan.md` 내용 확인
	```

	1-3 **Validation Planning**
	- 관련 메뉴얼 : `_manuals/_plan.md`, `_manuals/_validate.md`
	```
	 - `Phase {i}` 수행 후 결과 품질 평가를 위한 평가 방안 및 기준 수립, 그 결과를 `_manuals/workflow/Phase{i}_Validate.md`파일로 문서화
	```

	1-4 **Create Files**
	```
	 - `Phase {i}` 수행을 위한 폴더 및 파일 목록 제안 및 생성 
	```

2. **Phase task 수행 : Execution**
	- 관련 메뉴얼 : `_manuals/_execute.md`
	```
	- 반드시 `_manuals/workflow/Phase{i}_plan.md`에 기반하여 Task수행 
	- `_manuals/workflow/Phase{i}_plan.md`내 `CheckList`를 하나씩 확인해가며 Task 수행
	- 이전 Phase에서 코드 변경이 필요한 경우 `User`에게 내용을 반드시 확인시키고 승인을 얻은 뒤 코드 변경 수행
	```

3. **Phase task 수행결과 점검 : Validataion**
	- 관련 메뉴얼 : `_manuals/_validate.md`
	```
	 - `_manuals/workflow/Phase{i}_Validate.md`에 의거하여 `Phase {i}` 수행결과 점검
	 - `_manuals/workflow/Phase{i}_plan.md`내 `CheckList` 수행결과 점검
	 - 수정 및 보완 필요사항에 대해 `_manuals/workflow/Phase{i}_Improve.md`파일로 저장
	```
 
4. **Phase task 수정 보완 및 개선 : Improvement**
	```
	 - `_manuals/workflow/Phase{i}_Validate.md`에 의거하여 `Phase{i}`에서 수행한 결과물에 대한 수정 및 보완 사항 반영
	 - 필요시 `Planning` 단계로 돌아가서 목표 달성시까지 Loop재실행
	 - 목표 변경이 필요시 반드시 `User`에게 확인한 뒤 변경
	 - 전체 Project 목표에 대한 변경이 필요한지 점검, 변경 필요시 `User`에게 변경 방향성에 대해 반드시 동의를 구하고 `_manuals/workflow/Project_Validate.md`파일에 변경내용 반영
	```

5. **Phase task 결과 보고 : Feedback**
	- 관련 메뉴얼 : `_manuals/_feedback.md`
	```
	- 해당 `Phase {i}`에서 수행한 모든 내용(Plan, Execute, Valid, Improve)에 대한 실행결과를 요약정리하여 `_reports/Phase{i}_report.md`로 저장
	- 각 실행결과에 반드시 포함되어야 할 사항
		- 각 수행에 대한 요약
		- 주요결정사항 및 그이유
		- 금번 task에서 수행한 내용 중 다음 task와 연계되는 사항에 대한 정리
	```
## MVP Scope
- Streamlit을 이용한 단일 페이지 구성 (`app/` 폴더내 작성)

## Phase 1. `MVP 1.` VectorDB 구성
- 상세 내용 : `_manuals/project_requirements.md` 파일 내 `# MVP1` 파트 확인
- `data/`내 파일들을 이용하여 `Phase 2`, `Phase 3`에서 LLM 요약 및 ChatBot 서비스 구현을 위한 VectorStoreDB 생성, RAG index 생성
	- `data/`내에 있는 파일을 읽고 embedding하여 `data/vectorstore/` 폴더를 생성하여 vector저장소(Chroma DB 등)에 저장
  	- PDF는 chunking/embedding 후 vector store에 저장한다. (`_manuals/data.md` 참고)
  	- CSV는 구조 분석 후, 필요 시 텍스트 요약 또는 레코드 문서화하여 vector store에 저장한다.
  	- 집계/비교/추세/조건 필터링 중심 질문은 pandas 경로를 우선 사용하고, 설명/원인/특징/요약 중심 질문은 RAG 또는 LLM 요약 경로를 우선 사용한다.

## Phase 2. `MVP 2.` Streamlit DashBoard구성
- 상세 내용 : `_manuals/project_requirements.md` 파일 내 `# MVP2` 파트 확인
- App service를 위한 `app/main.py`를 생성하고, Streamlit을 이용하여 전체 Layout 구성
- `data/`내에 있는 파일을 읽고 요약정보를 시각화 Dashboard 생성 
   	- csv 파일 load 
   	- 중요 Feature Group별 interactive 통계분석 및 통계 Chart (MultiSelect Filter 조건 활용)
	- 년도별 insight 및 해당 년도 Claim/VOC 주요 특징에 대해 요약내용 서술 (LLM활용)
	- 중요 Feature Group별 insight 및 요약 내용 서술 (LLM 활용 실시간 요약글 생성)
	
## Phase 3. `MVP 3.` ChatBot 구현
- 상세 내용 : `_manuals/project_requirements.md` 파일 내 `# MVP3` 파트 확인
  - 사용자가 텍스트로 질문하면 `data/`내 `vectorDB`를 검색하여 답변하는 Chatbot UI 구현, session이 유지중에는 대화내용 history를 기억하여 답변하도록 구성
   	- `vectorDB` 및 `RAG`를 이용
   	- `memory`를 이용하여 session이 유지되는 동안 답변내용을 저장하여 후속 대화에서 기억하도록 구현
	
## Phase 4. 실행 테스트, 개선 및 리팩토링
- 실행 테스트 (`streamlit run app/main.py`) 및 오류 수정.
- `_manuals/workflow/Project_Validate.md`파일에 의거하여 Project 수행결과 평가 
- 추가 개선 및 최적화 요소 검증, `User`에게 제안
- 리팩토링 및 `reports/`폴더내 최종 리뷰보고서 작성(`Project_Final_Results.md`)
	

