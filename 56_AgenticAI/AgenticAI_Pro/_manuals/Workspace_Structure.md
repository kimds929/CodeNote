# Workspace Structure : 작업화면의 구조
	WorkSpace/
	├─ _archive/            # 백업용 폴더
    ├─ _manual/             # 세부 지시사항이 들어갈 폴더 (task별 markdown)
	├─ _reports/            # Agent의 task 이행결과 report를 보관하는 폴더
    ├─ _temp/               # 임시코드나 임시문서가 저장되는 폴더
	├─ app/                 # App서비스를 위한 Application
	├─ data/                # (필수) 이용가능한 documents, files, database(RDBMS, VectorDB 등)를 저장하는 폴더
	├─ tools/               # (필수) task이행을 위한 코드 및 tool들이 저장되는 폴더
	├─ AGENTS.md            # (필수) 프로젝트 전체의 Top Manual


# 주요 폴더내 구조
## `_manual/` : agent실행시 세부 지침을 담고 있는 폴더
- `_manual/Workspace_Structure.md` : Workspace내 전체 구조 및 주요 파일에 대한 설명자료
- `_manual/environments` : 환경 세팅, Library버전, LLM module, embedding module 정보 등을 담고 있는 자료

- `_manual/_plan.md` : task실행 전 planning과 관련된 manual
- `_manual/_execute.md` : Task 실행과 관련된 manual, Code 작성, 변경, 디버깅과 관련된 rule
- `_manual/_validation.md` : task실행 후 검증에 관한 manual
- `_manual/_feedback.md` : Agent실행 후 User에게 Feedback방식과 관련된 manual
- `_manual/tabular.md` : tabular data를 handling하는데 사용되는 manual
- `_manual/data.md` : 사용 가능 data에 대한 정보를 담고 있는 파일
- `_manual/environments/envs.md` : 코드 환경, 사용가능한 Library 버전정보
- `_manual/environments/LLM_API_Guide.md` : 
