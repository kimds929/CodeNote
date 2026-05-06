# Planning Manual (`_plan.md`)

## 1. Purpose (목적)
이 문서는 Agent가 작업을 시작하기 전, **수행 방향성을 확립하고 작업 계획 및 검증 계획을 수립할 때 준수해야 하는 계획 수립 메뉴얼**이다.
Agent는 코드를 작성하거나 파일을 수정하기 전에, 현재 작업의 목표와 범위, 대상 파일, 실행 순서, 검증 기준, 리스크에 대한 계획을 수립하고 문서화 해야한다.

이 문서의 목적은 다음과 같다.
- 현재 Phase의 목표와 Scope(작업 범위)를 명확히 한다.
- 추상적인 목표를 실행 가능한 구체적 Task 단위로 분해한다.
- 생성 및 수정 대상 파일을 사전에 정의하여 구조적 충돌을 방지한다.
- 작업 결과를 검증 할 수 있는 계획을 수립하여 결과물의 품질을 담보한다.
- 불명확한 요구사항과 리스크를 사전에 식별하여 `User`와 조율한다.

---
## 2. Scope of This Manual
### 2.1. 이 문서가 다루는 것
이 문서는 아래 내용을 다룬다.
- Planning이 필요한 시점
- Planning 수행 전 확인해야 할 입력자료
- Planning 기본 원칙
- Planning 수행 절차
- Planning 산출물 형식
- Validation Plan 작성 기준
- Planning 품질 점검 기준
- 금지되는 Planning 패턴
- 다음 단계 실행을 위한 handoff 기준

### 2.2. 이 문서가 직접 결정하지 않는 것
이 문서는 아래 항목을 직접 결정하지 않는다.
- 프로젝트별 최종 목표
- 프로젝트별 우선순위
- 기술 스택 선택
- 특정 라이브러리 사용 여부
- 특정 파일 경로 또는 폴더 구조
- 도메인별 비즈니스 로직
- 보안 정책의 세부 내용
- 사용자 경험 또는 화면 디자인의 최종 방향

위 항목은 프로젝트별 문서에서 정의되어야 하며, Agent는 Planning 시 해당 문서를 함께 참고해야 한다.

---

## 3. Planning Role in the Agent Workflow
- Agent는 항상 아래 흐름을 따른다.
    ```Planning → Execution → Validation → Improvement → Feedback```
- Planning은 이 전체 흐름의 시작 단계다.
- Planning 단계에서는 아직 본격적인 구현을 수행하지 않는다.
- 단, 현재 상태를 파악하기 위한 파일 목록 확인, 문서 확인, 데이터 스키마 확인, 환경 확인 등은 허용된다.
- Planning의 핵심 역할은 다음과 같다.
    ```
    무엇을 할지 정한다.
    어디까지 할지 정한다.
    무엇을 만들지 정한다.
    어떻게 검증할지 정한다.
    무엇을 사용자에게 확인받아야 하는지 정한다.
    ```

---
## 4. When Planning is Required
아래 경우에는 반드시 Planning을 먼저 수행한다.
1. 새로운 Phase를 시작할 때
2. 새로운 주요 기능을 구현할 때
3. 현재 Phase의 범위가 변경될 가능성이 있을 때
4. 주요 설계 또는 파일 구조 변경이 필요할 때
5. 이전 Phase 산출물을 수정해야 할 가능성이 있을 때
6. 구현 전에 사용자 확인이 필요한 선택지가 존재할 때
7. 새로운 데이터 파일을 사용해야 할 때
8. 새로운 라이브러리 또는 API 사용이 필요할 때
9. 에러 수정이 현재 Phase 범위를 넘을 가능성이 있을 때

---

## 5. Planning Inputs
Planning 수행 전 Agent는 아래 자료를 반드시 확인해야 한다.

### 5.1. Required Manuals (필수 확인 문서)
- `AGENTS.md`
- `_manuals/project_requirements.md`
- `_manuals/Workspace_Structure.md`
- `_manuals/envs.md`
- `_manuals/data.md`
- `_manuals/_validate.md`
- `_manuals/_execute.md`
- `_manuals/_feedback.md`

### 5.2. Previous Outputs (작업 이력 및 산출물)
- `_reports/` 내 최신 `.md` 보고서 (존재 시)
- `_manuals/workflow/Phase{i}_plan.md` (존재 시)
- `_manuals/workflow/Phase{i}_Validate.md` (존재 시)
- `_manuals/workflow/Phase{i}_Improve.md` (존재 시)
- `_manuals/workflow/Project_Validate.md` (존재 시)

### 5.3. Actual Workspace (실제 작업 대상)
- 현재 `Workspace` 폴더 구조
- `data/` 폴더 내 실제 파일 목록
- 현재 Phase에서 사용 가능한 기존 코드 및 문서
- Code 실행 환경 및 Library 정보 (`_manuals/envs.md`)

### 5.4 입력자료 부족 시 원칙
필수 자료가 없거나 접근할 수 없는 경우 아래 원칙을 따른다.
- 임의로 가정하지 않는다.
- 확인된 사실과 확인되지 않은 사실을 구분한다.
- 현재 Planning에 필요한 최소 사실만 정리한다.
- 부족한 정보는 Risks / Open Questions에 기록한다.
- 사용자 확인이 필요한 경우 선택지를 함께 제시한다.


---
## 6. Core Principles (계획 수립 기본 원칙)
6-1. **목표 명확화 (Goal Clarity)**
- 현재 작업의 목표를 명확히 정의한다.
- 목표는 결과 중심으로 작성한다.
- 목표는 가능하면 1~3줄로 요약한다.
- 목표가 모호하면 구현 전에 사용자에게 확인한다.

6-2. **범위 통제 (Scope Control)**
- 현재 Task의 목표에만 집중한다. 현재 이후의 기능은 절대 현재 계획에 포함하지 않는다.
- 다음 Phase와의 연결 구조는 설계 제안 수준으로만 다룬다.
- `In-Scope`(해야 할 일)와 `Out-of-Scope`(하지 말아야 할 일)를 명확히 구분한다.

6-3. **실행 가능한 세분화 (Actionable Breakdown)**
- 계획은 실제로 실행 가능한 단위로 분해되어야 한다.
- Task는 "데이터 로드", "UI 구성"처럼 추상적으로 적지 않는다.
- 가능하면 파일 단위, 함수 단위, 모듈 단위, UI 컴포넌트 단위, 테스트 단위로 분해한다.
- 가능하면 각 작업은 완료 여부를 판단할 수 있어야 한다.
- "A 파일을 읽어 B 함수로 전처리한 뒤 C 파일에 저장한다" 수준으로 구체화하여 Checklist를 작성한다.


6-4. **의존성 파악 (Dependency Check)**
- 해당 Phase를 수행하기 위해 필요한 데이터(`data/`), 환경(`envs.md`), 이전 Phase의 산출물(`_reports/`)이 모두 준비되었는지 먼저 확인한다.
- Planning 단계에서는 현재 작업에 필요한 의존성을 사전에 확인해야 한다. 확인 대상은 다음과 같다.
    - 필요한 데이터 파일
    - 필요한 기존 코드
    - 필요한 문서
    - 필요한 라이브러리
    - 필요한 환경 변수
    - 필요한 API 또는 인증 정보 존재 여부
    - 이전 Phase 산출물
    - 현재 작업에서 참조해야 할 manual

  단, 보안 규칙에 따라 .env, 인증키, 계정 정보 등 민감한 파일은 열람하거나 노출하지 않는다.
- 의존성에 문제가 있거나 파일이 존재하지 않으면, 임의로 진행하지 말고 `User`에게 보고한다.

6-5. **검증 가능성 (Validation Readiness)**
- 계획은 반드시 검증 가능해야 한다.
- 각 주요 작업은 완료 여부를 확인할 수 있는 테스트 또는 검증 항목 및 기준과 연결되어야 한다.
- 가능하다면 정성적 평가기준보다는 정량적인 기준을 바탕으로 검증계획을 수립하고 결과를 검증한다.
- Planning 단계에서 최소한의 Validation Plan을 함께 작성한다.

6-6. **리스크 및 질문 식별 (Risk & Question)**
- 요구사항이 불명확하거나, 두 가지 이상의 구현 방식이 존재할 경우 임의로 결정하지 않는다.
- 계획 문서 하단에 `User`에게 물어볼 질문(선택지 포함)을 명시한다.
- 다음 상황에서는 `User`의 확인이 필요하다.
    - 두 가지 이상의 구현 방식이 존재하는 경우
    - 데이터 구조가 예상과 다른 경우
    - 현재 Phase 범위를 벗어나는 구현이 필요한 경우
    - 외부 라이브러리 추가 설치가 필요한 경우
    - 기존 파일 구조를 크게 바꿔야 하는 경우
    - 데이터 전처리 기준을 결정해야 하는 경우
    - 비용이 발생할 수 있는 API 사용이 필요한 경우

6-7. **실데이터 및 사실 기반 계획 수립 (No Fake Assumptions)**
- 존재하지 않는 데이터, 파일, API, 환경을 전제로 계획하지 않는다.
- 확인되지 않은 사항을 사실처럼 쓰지 않는다.
- Mock Data, Dummy Data, Placeholder를 전제로 핵심 계획을 세우지 않는다.
- 프로젝트 규칙상 mock data가 명시적으로 허용된 경우에만 사용할 수 있으며, 그 사실을 문서에 기록한다.
- 데이터가 없으면 "데이터 필요"라고 보고한다.
- 테스트용 데이터가 필요한 경우에도 임의 생성하지 않고 `Ueer` 확인을 받는다.

6-8. **문서 일관성 (Documentation Consistency)**
- 계획 문서는 실제 구현과 일치해야 한다.
- 계획이 바뀌면 관련 문서도 갱신해야 한다.
- 문서와 코드가 따로 움직이지 않도록 한다.
- 파일명, 경로, Phase 이름, Task 이름은 일관되게 사용한다.

6-9. **점진적 확장 (Minimal First, Expand Later)**
- 처음부터 완성형 구조를 만드는 계획을 세우지는 않는다.
- Agent는 항상 다음 순서를 우선한다 : 작게 동작하는 구조 → 검증 → 개선 → 확장
- 따라서 계획 수립 시에는 다음 원칙을 따른다.
    - 한 Phase 안에서도 최소 실행 단위부터 만든다.
    - 복잡한 최적화는 마지막 개선 단계에서 다룬다.
    - LangGraph, Multi-Agent, AgentExecutor 등 고급 구조는 MVP가 정상 동작한 뒤 도입한다.



---
## 7. Planning Process (계획 수립 절차)
### Step 1. Current Work 정의
현재 요청이 어떤 작업 단위인지 정의한다.
```
예:
 - Project
 - Phase
 - Task
 - Feature
 - Bugfix
 - Refactoring
 - Validation
 - Documentation

정의할 내용:
 - 현재 작업 이름
 - 현재 Phase 번호 또는 이름
 - 현재 작업의 목적
 - 현재 작업이 전체 workflow에서 차지하는 위치
```

### Step 2. Context 수집
다음 자료를 확인한다.
 - `AGENTS.md`
 - 프로젝트 요구사항 문서 (`_manuals/project_requirements.md`)
 - workspace 구조 문서 (`_manuals/Workspace_Structure.md`)
 - 데이터 문서 (`_manuals/data.md`)
 - 환경 문서 (`_manuals/envs.md`)
 - 최신 보고서 및 이전 계획 문서 (`_reports/*.md`)
 - 이전 검증 문서 (`Phase{i}_Validate.md`)
 - 기존 코드 및 산출물 (`tools/*`, `app/*`)
 - 에서 현재 Phase의 목표 확인 (`_manual/Phase{i}_plan.md`)

### Step 3. Goal 정리
이번 작업의 목표를 명확히 요약한다.
작성 기준:
 - 결과 중심으로 작성한다.
 - 너무 길게 쓰지 않는다.
 - 하나의 Phase에 여러 목표가 있다면 우선순위를 구분한다.

### Step 4. Scope 정의
현재 작업의 범위를 정의한다.
반드시 아래 두 가지를 구분한다.
```
 - In-Scope
    - 이번 작업에서 수행할 항목

 - Out-of-Scope
    - 이번 작업에서 수행하지 않을 항목
    - 다음 작업에서 수행할 항목
    - `User` 승인 전에는 수행하면 안 되는 항목
```

### Step 5. Dependency / Constraint 점검
현재 작업에 필요한 의존성과 제약사항을 확인한다.
작성 대상:
 - 필요한 데이터
 - 필요한 기존 산출물
 - 필요한 환경 / 라이브러리
 - 필요한 설정 파일
 - 외부 API 또는 서비스
 - 보안상 접근하면 안 되는 파일
 - 현재 확인되지 않은 정보




###########################################################################################






























- **Step 2. 파일 구조 설계**
  - 이번 Phase에서 생성하거나 수정해야 할 파일 목록(`app/`, `tools/`, `data/` 등) 도출
- **Step 3. Task Checklist 작성**
  - 구현해야 할 기능을 논리적 순서(데이터 로드 → 로직 구현 → UI 연결 등)에 따라 Step별로 분할
- **Step 4. 검증 계획 수립**
  - 이 Phase가 성공적으로 끝났음을 증명할 수 있는 최소한의 테스트 시나리오 도출
- **Step 5. 문서화**
  - 아래 제공된 템플릿에 맞춰 `_manuals/workflow/Phase{i}_plan.md` 및 `Phase{i}_Validate.md` 생성

---

## 4. Templates (산출물 표준 양식)

Agent는 계획 수립 결과를 반드시 아래의 Markdown 템플릿 구조에 맞춰 작성해야 한다.

### 4.1. `Phase{i}_plan.md` 템플릿
```markdown
# Phase {i} Plan: [Phase 이름]

## 1. Goal (목표)
- 이번 Phase에서 달성하고자 하는 핵심 목표 1~2줄 요약

## 2. Scope (작업 범위)
- **In-Scope (포함 대상):**
  - [ ] 기능 1
  - [ ] 기능 2
   ...

- **Out-of-Scope (제외 대상):**
  - [ ] 다음 Phase에서 할 일이나 현재 범위가 아닌 것

## 3. Target Files (대상 파일)
- **생성할 파일:**
  - `{경로}/{파일명}.py`: (역할 설명)
- **수정할 파일:**
  - `{경로}/{파일명}.py`: (수정 내용 설명)

## 4. Task Checklist (상세 작업 단계)
- [ ] **Step 1: [작업명]**
  - 세부 구현 내용 1
  - 세부 구현 내용 2
  ...
- [ ] **Step 2: [작업명]**
  - 세부 구현 내용 1
  - 세부 구현 내용 2
  ...

...

## 5. Open Issues & Questions (질문 및 확인 사항)
- [ ] (User 확인 필요) A 방식으로 구현할지, B 방식으로 구현할지?
- [ ] (리스크) 특정 데이터의 결측치 처리 방안





# Phase {i} Validation Plan

## 1. Validation Goal (검증 목표)
- 이 Phase의 결과물이 충족해야 하는 핵심 조건

## 2. Test Scenarios (테스트 시나리오)
| 시나리오 ID | 테스트 항목 | 입력/조건 | 기대 결과 (Expected Output) |
|---|---|---|---|
| TC-01 | 데이터 로드 | `data/sample.csv` 읽기 | 에러 없이 DataFrame 생성 및 shape 출력 |
| TC-02 | UI 표시 | `streamlit run` 실행 | 브라우저에 대시보드 타이틀 정상 표시 |

## 3. Acceptance Criteria (최소 통과 기준)
- [ ] 에러 로그 없이 코드가 정상 실행되는가?
- [ ] 요구사항에 명시된 핵심 기능이 동작하는가?
- [ ] Mock Data가 아닌 실제 `data/` 폴더의 데이터를 사용했는가?