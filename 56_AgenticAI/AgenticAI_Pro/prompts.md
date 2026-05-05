###########################################################################################
# Prompt Phase 1 ##########################################################################

Initialize this project using the workspace `AGENTS.md` as the global manual.

Follow these rules strictly:
- Only execute Phase 1 of the Workflow.
- Do not implement any code yet.
- Do not create, modify, or delete files before my approval.
- First inspect the workspace and available files.
- Then propose:
  1. current phase
  2. scope of work
  3. files to read
  4. files to create or update
  5. step-by-step plan

Focus on:
- checking the workspace structure
- verifying available data in `data/`
- scanning schema/content
- preparing `instructions/data_schema.md` and `instructions/project_plan.md`

Respond in Korean.


# Prompt Phase 2 ##########################################################################
현재 workspace의 `AGENTS.md`와 `instructions/project_plan.md`, `reports/`내의 `.md`파일을 최우선으로 읽고 현재 상황을 파악하세요.

이제 Workflow의 Phase 2 (MVP 1. VectorDB 구성) 작업을 시작합니다.
다른 Phase는 절대 진행하지 마세요.

요구사항:
1. `instructions/data_schema.md`를 참고하여 데이터 구조에 맞는 Loader를 설계하세요.
2. CSV는 전체를 VectorDB에 넣기보다, pandas를 이용한 분석용 Loader를 우선 고려하세요.
3. PDF는 chunking 및 Chroma DB 저장을 위한 로직을 설계하세요.

중요 규칙:
- 바로 코드를 작성하거나 파일을 생성하지 마세요.
- 먼저 Phase 2에서 생성/수정할 파일 목록(`src/` 및 `database/` 내)과 각 파일의 핵심 로직(설계안)을 제안하세요.
- 제안 후 저의 승인을 받으면 코딩을 시작하세요.



# Prompt Phase 3 ##########################################################################
현재 workspace의 `AGENTS.md`와 `instructions/project_plan.md`, `reports/`내의 `.md`파일을 최우선으로 읽고 현재 상황을 파악하세요.
또한 `instructions/data_schema.md`와 Phase 2에서 작성된 `src/` 및 `database/` 폴더의 코드들을 리뷰하여 데이터 구조와 로직을 이해하세요.

이제 Workflow의 **Phase 3 (MVP 2. Streamlit DashBoard 구성)** 작업을 시작합니다.
다른 Phase(특히 Phase 4 Chatbot)는 절대 진행하지 마세요.

[수행할 작업 범위]
- Step 9: `app/main.py` 생성 및 Streamlit 전체 Layout 구성 (왼쪽: Dashboard, 오른쪽: Chatbot 영역 확보)
- Step 10: CSV 데이터를 로드하여 년도별 주요 통계 및 데이터 분석결과 시각화 (왼쪽 영역)
- Step 11: 년도별 데이터 분석결과 및 insight에 대해 Text 요약 디스플레이

[중요 규칙]
1. **기능 분리:** 데이터 로드 및 분석 로직은 `app/main.py`에 전부 때려넣지 말고, `src/` 폴더 내에 별도 모듈(예: `src/dashboard_logic.py` 등)로 분리하여 `app/main.py`에서 import 하세요.
2. **실행 금지:** 코드를 작성하되, 아직 터미널에서 `streamlit run`을 실행하지 마세요. (실행 테스트는 Phase 5의 Step 15에서 진행합니다.)
3. **승인 후 코딩:** 바로 코드를 작성하지 마세요. 먼저 `app/main.py`의 UI 레이아웃 구조와 `src/`에 추가/수정할 파일의 설계안을 제안하세요.

제안을 확인한 후 제가 승인하면 코딩을 시작하세요. 응답은 한국어로 구조화해서 작성하세요.
