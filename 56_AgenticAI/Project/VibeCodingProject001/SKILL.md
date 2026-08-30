---
name: project_supervisor
description: ddd

---



# 스킬: project_supervisor

## 목적

## 정의
`project_supervisor`에서 명명되는 용어를 정의한다.
- `<REPO_ROOT>` : 현재 Workspace의 최상위 ROOT 경로를 의미한다.
- `<SKILL_ROOT>` : `project_supervisor`폴더의 경로를 의미한다. (즉, `SKILL.md`파일이 담겨있는 폴더의 경로)


## 절차
1. 프로젝트 진행상태 파악
- `<REPO_ROOT>/project/AGENTS.md` 존재 여부 확인
    --> |YES| 프로젝트가 진행중
    --> |NO| 신규 프로젝트 진행 

2. 신규프로젝트 진행
    2.1. 
        2.1.1. Capability & Environment Discovery
            - Repo 접근 가능 여부
            - Git 사용 여부
            - test / lint / typecheck / build 환경
            - CI/CD 존재 여부
            - Browser / Runtime / Visual 검증 가능 여부
            - MCP 또는 외부 도구 존재 여부
            - 프로젝트 문서 위치
            - Agent가 artifact를 쓸 수 있는 영역
        2.1.2. WorkSpace Structure 확인
            - `<REPO_ROOT>`의 절대 경로 확인
            - `<REPO_ROOT>`내 전체 폴더 및 파일 구조 확인
        2.1.3. 개발환경 확인
            - `Python Interpreter` 경로 및 `Python` 버전 확인
    
    2.2. `project` 폴더 생성
        - `<REPO_ROOT>/project` 폴더를 생성한다.

    2.3 `project


## RULE
 - 원칙적으로 `<REPO_ROOT>/project` 폴더내의 파일만 수정한다.




---
---
---
1. 먼저 묻는다: **"어떤 분이신가요?"** — (a) 회사원 (b) 연구원·대학원생 (c) 학부생 (d) 기타
2. 유형에 맞춰 **한 번에 하나씩** 묻는다 (전체 5~7분, 답이 짧아도 캐묻지 않기):
   - 공통 ①: 하는 일(연구·전공)을 한 문장으로 하면?
   - 공통 ②: 주로 **들어오는 자료**는? (회의록·이메일·데이터·논문·강의자료 등)
   - 공통 ③: 계속 쌓아두고 싶은 **지식 주제** 2~4개는?
   - 유형별 ④: 일(연구·공부)은 **어떤 단위로 움직이나**?
     - 회사원 — 회의 · 프로젝트 · 거래처 · 보고 주기
     - 연구원·대학원생 — 논문 · 실험 · 과제 · 랩미팅
     - 학부생 — 수업 · 과제 · 팀플 · 시험
   - 공통 ⑤: 주로 만드는 **산출물**은? (보고서·논문·발표자료·과제 등)
   - 마무리 ⑥: 문서 **말투·양식** 선호는? / AI가 **하지 말았으면** 하는 것 하나?
3. 답변 요약을 보여주고 확인받는다.
4. 확인받으면:
   - `내 프로필.md`를 Vault 최상위에 저장 — 인터뷰 답 정리 (유형·하는 일·인풋·지식 주제·일의 단위·산출물)
   - `templates/AGENTS-structure.md` 구조 그대로 `AGENTS.md`를 Vault 최상위에 생성 — ①나는 누구 ③작성 규칙 ④하지 말 것을 인터뷰 답으로 채우고, ②폴더 구조는 뼈대 기본 설명 유지

## 출력

- `내 프로필.md` · `AGENTS.md` (둘 다 Vault 최상위)

## 규칙



## 검증 기준

- 원본에 없는 내용을 지어내지 않았는가
- Action Item에 담당자가 다 붙어 있는가
- 핵심 요약이 3줄 이내인가
