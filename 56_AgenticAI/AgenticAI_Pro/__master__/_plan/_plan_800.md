# Planning Manual (`_plan.md`)

## 1. Purpose

이 문서는 `AGENTS.md`와 함께 사용하는 범용 Planning Manual이다.  
Agent가 어떤 프로젝트에서든 작업을 시작하기 전에 **무엇을 할지, 어디까지 할지, 어떤 파일을 만들거나 수정할지, 어떻게 검증할지**를 먼저 정리하도록 돕는다.

이 문서는 완성형 규칙집이 아니라, 실제 프로젝트를 진행하면서 점진적으로 보완해나가는 **초기 실행형 초안**이다.

이 문서의 목적은 다음과 같다.

- 현재 작업의 목표와 범위를 명확히 한다.
- 추상적인 요청을 실행 가능한 Task로 분해한다.
- 생성/수정/검토할 파일을 사전에 정의한다.
- 현재 작업이 전체 Workflow에서 어디에 위치하는지 확인한다.
- 검증 가능한 계획을 세운다.
- 불확실한 요구사항과 리스크를 사전에 식별한다.
- 다음 단계인 Execution으로 안전하게 넘길 수 있는 기준을 만든다.

---

## 2. Role in Agent Workflow

Agent는 기본적으로 아래 흐름을 따른다.

```text
Planning → Execution → Validation → Improvement → Feedback
```

Planning은 이 흐름의 첫 단계다.

Planning 단계에서는 본격적인 구현을 시작하지 않는다.  
다만 현재 상태 파악을 위한 아래 작업은 허용된다.

- workspace 구조 확인
- 문서 확인
- 데이터 파일 목록 확인
- 데이터 스키마 요약 확인
- 환경 및 라이브러리 확인
- 이전 보고서 확인
- 기존 코드 구조 확인

Planning의 핵심 질문은 다음과 같다.

```text
1. 지금 무엇을 해야 하는가?
2. 어디까지 해야 하는가?
3. 무엇은 하지 말아야 하는가?
4. 어떤 파일을 만들거나 수정해야 하는가?
5. 어떻게 완료 여부를 검증할 것인가?
6. 사용자 확인이 필요한 것은 무엇인가?
```

---

## 3. When Planning is Required

아래 경우에는 반드시 Planning을 먼저 수행한다.

1. 새로운 프로젝트를 시작할 때
2. 새로운 Phase를 시작할 때
3. 새로운 주요 기능을 구현할 때
4. 기존 코드나 문서 구조를 변경해야 할 때
5. 현재 작업 범위가 변경될 가능성이 있을 때
6. 이전 Phase 산출물을 수정해야 할 가능성이 있을 때
7. 새로운 데이터 파일을 사용해야 할 때
8. 새로운 라이브러리, API, 외부 도구 사용이 필요할 때
9. 사용자 확인이 필요한 선택지가 존재할 때
10. 에러 수정이 현재 Phase 범위를 넘어설 가능성이 있을 때

단순 오타 수정, 주석 보완, 매우 작은 문서 수정처럼 위험도가 낮은 작업은 간단한 계획 요약만으로 진행할 수 있다.

---

## 4. Required Inputs

Planning 수행 전 Agent는 가능한 범위에서 아래 자료를 확인한다.

### 4.1. Required Project Documents

프로젝트에 존재하는 경우 아래 문서를 확인한다.

- `AGENTS.md`
- `_manuals/project_requirements.md`
- `_manuals/Workspace_Structure.md`
- `_manuals/envs.md`
- `_manuals/data.md`
- `_manuals/_execute.md`
- `_manuals/_validate.md`
- `_manuals/_feedback.md`

### 4.2. Previous Outputs

존재하는 경우 아래 자료를 확인한다.

- `_reports/` 내 최신 보고서
- `_manuals/workflow/` 내 이전 plan 문서
- `_manuals/workflow/` 내 이전 validation 문서
- 이전 Phase에서 생성된 코드, 문서, 설정 파일

### 4.3. Actual Workspace

Planning은 실제 workspace 기준으로 수행한다.

확인 대상은 다음과 같다.

- 현재 폴더 구조
- 현재 존재하는 코드 파일
- 현재 존재하는 문서 파일
- 현재 존재하는 설정 파일
- `data/` 폴더 내 실제 파일 목록
- 현재 Phase에서 사용할 수 있는 기존 모듈
- 실행 환경 및 라이브러리 상태

### 4.4. If Required Inputs Are Missing

필수 자료가 없거나 확인할 수 없는 경우 아래 원칙을 따른다.

- 임의로 가정하지 않는다.
- 확인된 사실과 확인되지 않은 사실을 구분한다.
- 부족한 정보는 `Risks / Open Questions`에 기록한다.
- 사용자 확인이 필요한 경우 선택지를 제시한다.
- 현재 작업을 진행할 수 없는 수준이면 구현을 시작하지 않는다.

---

## 5. Core Principles

### 5.1. Goal Clarity

- 현재 작업의 목표를 결과 중심으로 정의한다.
- 목표는 가능하면 1~3줄로 작성한다.
- 목표가 모호하면 구현 전에 사용자에게 확인한다.

좋지 않은 예:

```text
대시보드를 만든다.
```

좋은 예:

```text
`data/AA.csv`를 로드하여 연도별 주요 지표를 계산하고, Streamlit 화면에 통계 카드와 차트로 표시한다.
```

---

### 5.2. Scope Control

- 현재 Phase 또는 Task의 범위에 집중한다.
- 반드시 `In-Scope`와 `Out-of-Scope`를 구분한다.
- 다음 Phase의 구현은 현재 계획에 포함하지 않는다.
- 다음 Phase와의 연결 구조는 설계 메모 수준으로만 다룬다.
- 현재 범위를 넘어서는 구현이 필요하면 사용자 확인을 받는다.

---

### 5.3. Actionable Breakdown

- 계획은 실제 실행 가능한 단위로 분해한다.
- 추상적인 표현만 사용하지 않는다.
- 가능하면 파일 단위, 함수 단위, 모듈 단위, UI 컴포넌트 단위, 테스트 단위로 나눈다.
- 각 작업은 완료 여부를 판단할 수 있어야 한다.

좋지 않은 예:

```text
RAG 만들기
챗봇 구현
대시보드 개발
```

좋은 예:

```text
`tools/loaders.py`에 CSV 로드 함수를 작성한다.
`tools/vectorstore.py`에 Chroma 저장 함수를 작성한다.
`app/main.py`에 Streamlit sidebar 필터 영역을 연결한다.
```

---

### 5.4. Dependency Check

Planning 단계에서는 현재 작업에 필요한 의존성을 먼저 확인한다.

확인 대상은 다음과 같다.

- 필요한 데이터 파일
- 필요한 기존 코드
- 필요한 문서
- 필요한 라이브러리
- 필요한 실행 환경
- 필요한 API 또는 외부 도구
- 이전 Phase 산출물
- 현재 작업에서 참조해야 할 manual

보안 규칙에 따라 `.env`, 인증키, 계정 정보 등 민감한 파일은 열람하거나 노출하지 않는다.

---

### 5.5. Validation Readiness

- 모든 계획은 검증 가능해야 한다.
- 각 핵심 작업은 검증 항목과 연결되어야 한다.
- Planning 단계에서 최소한의 Validation Plan을 함께 작성한다.
- 검증 기준은 가능하면 실행 결과, 파일 존재 여부, 출력 형태, 테스트 통과 여부처럼 확인 가능한 방식으로 작성한다.

---

### 5.6. User Confirmation First

아래 상황에서는 Agent가 임의로 결정하지 않는다.

- 요구사항이 모호한 경우
- 구현 방식이 2개 이상이고 장단점이 있는 경우
- 데이터 구조 변경이 필요한 경우
- 외부 의존성 추가가 필요한 경우
- 기존 파일 구조를 크게 바꿔야 하는 경우
- 현재 Phase 범위를 벗어나는 구현이 필요한 경우
- 비용이 발생할 수 있는 API 호출이 필요한 경우

이 경우 선택지와 장단점을 정리하고 사용자 확인을 요청한다.

---

### 5.7. No Fake Assumptions

- 존재하지 않는 데이터, 파일, API, 환경을 전제로 계획하지 않는다.
- 확인되지 않은 사항을 사실처럼 쓰지 않는다.
- Mock Data, Dummy Data, Placeholder를 핵심 계획의 전제로 삼지 않는다.
- 테스트 데이터가 필요한 경우에도 프로젝트 규칙 또는 사용자 승인을 따른다.

---

### 5.8. Minimal First, Expand Later

- 처음부터 완성형 구조를 계획하지 않는다.
- 항상 작은 단위로 동작하는 구조를 먼저 만든다.
- 이후 검증 결과를 바탕으로 개선하고 확장한다.
- LangGraph, Multi-Agent, AgentExecutor 같은 고급 구조는 MVP가 정상 동작한 뒤 도입한다.

기본 순서:

```text
작게 동작하는 구조 → 검증 → 개선 → 확장
```

---

## 6. Planning Procedure

Agent는 작업 시작 시 아래 순서로 Planning을 수행한다.

### Step 1. Current Work 정의

현재 요청이 어떤 작업 단위인지 정의한다.

예:

- Project
- Phase
- Task
- Feature
- Bugfix
- Refactoring
- Validation
- Documentation

정리할 내용:

- 현재 작업 이름
- 관련 Phase
- 현재 작업의 목적
- 전체 Workflow에서의 위치

---

### Step 2. Context 확인

아래 내용을 확인한다.

- `AGENTS.md`
- 프로젝트 요구사항 문서
- workspace 구조 문서
- 데이터 문서
- 환경 문서
- 최신 보고서
- 이전 계획 문서
- 기존 코드 및 산출물

확인한 내용은 계획 문서의 `Context Summary`에 요약한다.

---

### Step 3. Goal 정리

이번 작업의 목표를 1~3줄로 정리한다.

작성 기준:

- 결과 중심으로 쓴다.
- 너무 길게 쓰지 않는다.
- 여러 목표가 있다면 우선순위를 구분한다.

---

### Step 4. Scope 정의

현재 작업의 범위를 정의한다.

반드시 아래 두 가지를 구분한다.

```markdown
### In-Scope
- 이번 작업에서 수행할 항목

### Out-of-Scope
- 이번 작업에서 수행하지 않을 항목
- 다음 Phase에서 수행할 항목
- 사용자 승인 전에는 수행하면 안 되는 항목
```

---

### Step 5. Dependency / Constraint 점검

현재 작업에 필요한 의존성과 제약사항을 확인한다.

작성 대상:

- 필요한 데이터
- 필요한 기존 산출물
- 필요한 환경 / 라이브러리
- 필요한 설정 파일
- 외부 API 또는 서비스
- 접근하면 안 되는 파일 / 정보
- 현재 확인되지 않은 정보

---

### Step 6. Target Outputs 도출

이번 작업의 결과로 생성, 수정, 검토되어야 하는 대상을 정의한다.

아래 세 가지로 구분한다.

```markdown
### To Create
- 새로 생성할 파일 또는 산출물

### To Modify
- 수정할 파일 또는 산출물

### To Review / Validate
- 검토하거나 검증할 파일 또는 산출물
```

각 항목에는 파일 경로와 역할을 함께 작성한다.

---

### Step 7. Task Breakdown

작업을 실행 가능한 단위로 분해한다.

작성 기준:

- 논리적 순서대로 작성한다.
- checklist 형태로 작성한다.
- 각 Step은 너무 크지 않아야 한다.
- 각 Step은 완료 여부를 판단할 수 있어야 한다.
- 코드 작업은 파일명, 함수명, 모듈명을 가능한 구체적으로 작성한다.
- 데이터 작업은 입력 파일, 처리 방식, 출력 결과를 명시한다.
- UI 작업은 표시 위치, 사용자 입력 방식, 출력 화면을 명시한다.

---

### Step 8. Validation Planning

현재 작업이 완료되었음을 어떻게 확인할지 정의한다.

포함 항목:

- 검증 목표
- 검증 항목
- 테스트 시나리오
- 기대 결과
- 최소 통과 기준
- 실패 시 처리 방식

가능하면 별도의 Validation Plan 문서로 저장한다.

---

### Step 9. Risks / Open Questions 정리

아래 항목을 분리하여 작성한다.

- 리스크
- 불확실성
- 사용자 확인 필요사항
- 현재 작업 범위 내에서 해결 가능한 문제
- 현재 작업 범위 밖이라 사용자 승인이 필요한 문제

사용자 확인이 필요한 항목은 가능하면 선택지와 장단점을 함께 제시한다.

---

### Step 10. Planning 문서화

Planning 결과를 프로젝트 표준 경로에 저장한다.

Phase 기반 프로젝트 권장 경로:

```text
_manuals/workflow/Phase{i}_plan.md
_manuals/workflow/Phase{i}_Validate.md
```

Task 기반 프로젝트 권장 경로:

```text
_manuals/workflow/Task_{task_name}_plan.md
_manuals/workflow/Task_{task_name}_Validate.md
```

프로젝트에서 다른 naming convention을 지정한 경우 해당 규칙을 따른다.

---

## 7. Required Outputs

Planning 결과로 아래 산출물이 생성되어야 한다.

### 7.1. Required

- Plan Document
- Validation Plan Document

### 7.2. Optional

필요 시 아래 산출물을 함께 제안하거나 생성할 수 있다.

- Improve Plan
- File Structure Proposal
- Decision Memo
- Risk Memo
- User Question List

### 7.3. Plan Document에 포함할 항목

- 현재 작업 단위
- 목표
- Context Summary
- In-Scope
- Out-of-Scope
- 의존성 / 제약사항
- 생성 / 수정 / 검토 대상
- 상세 작업 단계
- 검증 계획 링크
- 리스크 / 질문사항
- 완료 기준

### 7.4. Validation Plan Document에 포함할 항목

- 검증 목표
- 검증 항목
- 테스트 시나리오
- 기대 결과
- 최소 통과 기준
- 실패 처리 방식

---

## 8. Generic Plan Document Template

Agent는 계획 문서를 아래 구조에 맞춰 작성하는 것을 기본 원칙으로 한다.

```markdown
# [Work Unit Name] Plan

## 1. Work Unit
- Type: Project / Phase / Task / Feature / Bugfix / Refactoring / Validation / Documentation
- Name:
- Related Phase:
- Current Status:

## 2. Goal
- 이번 작업의 핵심 목표를 1~3줄로 요약

## 3. Context Summary
- 참고한 문서:
- 참고한 이전 보고서:
- 확인한 workspace 상태:
- 확인한 데이터 / 입력 자산:

## 4. Scope

### In-Scope
- [ ] 현재 작업에서 수행할 항목 1
- [ ] 현재 작업에서 수행할 항목 2
- [ ] 현재 작업에서 수행할 항목 3

### Out-of-Scope
- [ ] 현재 작업 범위 밖의 항목
- [ ] 다음 Phase 또는 별도 승인 후 진행할 항목
- [ ] 현재 작업에서 다루지 않을 항목

## 5. Dependencies / Constraints
- 필요한 입력자료:
- 필요한 기존 산출물:
- 필요한 환경 / 도구:
- 확인된 제약사항:
- 접근하면 안 되는 파일 / 정보:

## 6. Target Outputs

### To Create
- `path/or/name`: 역할 설명

### To Modify
- `path/or/name`: 수정 목적 설명

### To Review / Validate
- `path/or/name`: 검토 목적 설명

## 7. Detailed Task Checklist
- [ ] Step 1. [작업명]
  - 세부 작업 1
  - 세부 작업 2

- [ ] Step 2. [작업명]
  - 세부 작업 1
  - 세부 작업 2

- [ ] Step 3. [작업명]
  - 세부 작업 1
  - 세부 작업 2

## 8. Validation Link
- 관련 검증 문서: `path/or/name`

## 9. Risks / Open Questions

### Risks
- [ ] 리스크 1
- [ ] 리스크 2

### Open Questions
- [ ] 사용자 확인 필요사항 1
- [ ] 사용자 확인 필요사항 2

### Options for User Decision
| 항목 | 선택지 | 장점 | 단점 | 추천 |
|---|---|---|---|---|
| 예: 저장 위치 | A안 |  |  |  |
| 예: 저장 위치 | B안 |  |  |  |

## 10. Exit Criteria
- [ ] 핵심 목표가 충족되었는가
- [ ] 최소 검증 기준을 만족하는가
- [ ] 문서와 구현이 일관되는가
- [ ] 현재 범위를 벗어난 구현이 포함되지 않았는가
```

---

## 9. Generic Validation Plan Template

Agent는 검증 계획 문서를 아래 구조에 맞춰 작성한다.

```markdown
# [Work Unit Name] Validation Plan

## 1. Validation Goal
- 이번 작업 결과물이 충족해야 하는 핵심 검증 목표를 요약

## 2. Validation Items
- [ ] 기능 검증
- [ ] 데이터 / 입력 검증
- [ ] 실행 검증
- [ ] 문서 일관성 검증
- [ ] 범위 준수 검증

## 3. Test Scenarios

| Test ID | 검증 항목 | 입력 / 조건 | 기대 결과 |
|---|---|---|---|
| TC-01 | 입력 로드 | 실제 입력 자산 사용 | 오류 없이 로드됨 |
| TC-02 | 핵심 기능 실행 | 정상 입력 제공 | 기대한 출력 생성 |
| TC-03 | 엔트리포인트 실행 | 실행 명령 수행 | 정상 실행됨 |

## 4. Acceptance Criteria
- [ ] 실행 또는 동작이 정상인가
- [ ] 현재 작업의 핵심 목표를 충족하는가
- [ ] 실제 입력 또는 승인된 자산을 사용했는가
- [ ] 현재 범위를 벗어난 구현을 포함하지 않았는가
- [ ] 문서와 구현이 일관되는가

## 5. Failure Handling
- 실패 시 원인을 기능 / 데이터 / 환경 / 설계 / 범위 문제로 분류한다.
- 현재 작업 범위 내 수정 가능 여부를 판단한다.
- 현재 범위를 벗어나는 경우 사용자에게 보고하고 선택지를 제시한다.
- 실패 원인과 수정 방향은 이후 Feedback 또는 Report 문서에 반영한다.
```

---

## 10. Planning Quality Checklist

Planning 문서를 작성한 후 Agent는 아래 항목으로 자체 점검한다.

```markdown
# Planning Quality Checklist

- [ ] 현재 작업 단위가 명확하게 정의되었는가
- [ ] 현재 Phase 또는 Task가 전체 workflow에서 어디에 위치하는지 명시되었는가
- [ ] Goal이 명확하고 결과 중심으로 작성되었는가
- [ ] In-Scope / Out-of-Scope가 분리되었는가
- [ ] 의존성과 제약사항이 기록되었는가
- [ ] 생성 / 수정 / 검토 대상이 구분되었는가
- [ ] 작업이 실행 가능한 수준으로 분해되었는가
- [ ] checklist 형태로 진행 순서가 표현되었는가
- [ ] 검증 계획 문서가 함께 정의되었는가
- [ ] 완료 기준이 존재하는가
- [ ] 사용자 확인 필요사항이 분리되었는가
- [ ] 확인되지 않은 가정을 사실처럼 쓰지 않았는가
- [ ] Mock Data 또는 Dummy Data를 전제로 하지 않았는가
- [ ] 현재 범위를 벗어난 항목이 계획에 섞이지 않았는가
- [ ] 보안상 열람하면 안 되는 파일을 참조하지 않았는가
- [ ] 문서와 실제 구현 방향이 일치하도록 설계되었는가
```

---

## 11. Prohibited Planning Patterns

아래와 같은 Planning은 금지한다.

### 11.1. 지나치게 추상적인 계획

금지 예:

```text
기능 구현
서비스 개발
시스템 고도화
RAG 만들기
대시보드 구현
```

이유:

- 실행 가능하지 않다.
- 완료 여부를 판단하기 어렵다.
- 생성 / 수정 대상 파일을 예측할 수 없다.

---

### 11.2. 범위 침범 계획

금지 예:

```text
현재 Phase가 VectorDB 구성인데 Chatbot UI까지 함께 구현한다.
현재 Task가 데이터 로드인데 Streamlit 화면 전체를 함께 만든다.
승인되지 않은 기능을 추가한다.
```

이유:

- Phase 제어가 무너진다.
- 일정과 품질 리스크가 증가한다.
- 사용자의 의사결정 없이 범위가 확장된다.

---

### 11.3. 의존성 무시 계획

금지 예:

```text
data/ 폴더 확인 없이 데이터 로더를 설계한다.
requirements.txt 또는 환경 문서 확인 없이 라이브러리를 추가한다.
기존 코드 구조 확인 없이 새 구조를 제안한다.
```

이유:

- 실제 실행 불가능할 가능성이 높다.
- 기존 구조와 충돌할 수 있다.

---

### 11.4. 검증 없는 계획

금지 예:

```text
구현 작업만 있고 실행 테스트 또는 검증 항목이 없다.
완료 기준 없이 작업 목록만 작성한다.
```

이유:

- 완료 여부를 객관적으로 판단할 수 없다.
- Validation 단계로 연결되지 않는다.

---

### 11.5. 사용자 확인 없는 임의 결정

금지 예:

```text
요구사항이 애매한데 Agent가 독단적으로 설계를 확정한다.
저장 위치, DB 선택, 외부 의존성 추가를 사용자 확인 없이 결정한다.
```

이유:

- 잘못된 방향으로 구현될 가능성이 높다.
- 이후 재작업이 발생할 수 있다.

---

### 11.6. 문서-구현 분리 계획

금지 예:

```text
계획 문서와 실제 구현 방향이 다를 것이 예상되는데도 계획을 수정하지 않는다.
파일명을 계획과 다르게 만들면서 문서를 갱신하지 않는다.
```

이유:

- 유지보수성과 추적 가능성이 떨어진다.
- 이후 Agent가 잘못된 문서를 기준으로 작업할 수 있다.

---

## 12. Recommended Planning Style

Agent는 Planning 문서를 작성할 때 아래 스타일을 따른다.

- 짧고 명확한 문장을 사용한다.
- 추상어보다 동작 중심 표현을 사용한다.
- 대상 경로, 파일, 산출물, 컴포넌트를 구체적으로 명시한다.
- checklist 중심으로 작업 순서를 표현한다.
- 리스크와 질문을 분리하여 정리한다.
- 검증 기준과 작업 단계를 연결한다.
- 사용자 의사결정이 필요한 항목은 선택지와 장단점을 함께 제시한다.
- 프로젝트별 세부 규칙은 별도 문서를 참조하도록 분리한다.
- 문서가 길어질 경우 표와 checklist를 활용한다.

---

## 13. Compatibility with Project-Specific Workflow

이 문서는 범용 Planning Manual이다.

프로젝트별 운영 방식과 함께 사용할 때는 아래 원칙을 따른다.

1. 프로젝트별 `AGENTS.md` 또는 동등한 Top Manual이 존재하면 그것을 함께 따른다.
2. 프로젝트가 특정 naming convention을 요구하면 그 형식을 사용한다.
3. 프로젝트가 특정 폴더 구조, 기술 스택, 보고 규칙, 검증 규칙을 정의했다면 Planning 문서에 반영한다.
4. 범용 원칙과 프로젝트별 규칙이 충돌하면 프로젝트별 상위 운영 문서의 우선순위 규칙을 따른다.
5. 단, 보안, 데이터 무결성, 사용자 확인 필요사항, 가짜 전제 금지 원칙은 유지한다.

---

## 14. Handoff to Execution

Planning이 완료되면 Agent는 Execution 단계로 넘어가기 전에 아래 내용을 사용자에게 제시한다.

```markdown
# Execution Handoff Summary

## Current Work
- 현재 작업 단위:
- 관련 Phase:

## Confirmed Scope
- 이번에 수행할 범위:

## Files to Create / Modify
- 생성할 파일:
- 수정할 파일:

## Validation Plan
- 검증 문서:
- 핵심 검증 기준:

## User Confirmation Needed
- [ ] 확인 필요사항 1
- [ ] 확인 필요사항 2

## Next Action
- 사용자 확인 후 `_manuals/_execute.md` 기준으로 실행을 시작한다.
```

사용자 확인이 필요한 항목이 없고, 상위 운영 문서에서 자동 실행이 허용된 경우에는 Execution 단계로 진행할 수 있다.

단, 아래 경우에는 반드시 사용자 확인을 받아야 한다.

- 현재 Phase 범위를 넘어서는 구현
- 외부 라이브러리 추가
- 데이터 구조 변경
- 저장 위치 변경
- 기존 설계 방향 변경
- 보안 또는 민감 정보와 관련된 작업
- 비용이 발생할 수 있는 API 호출

---

## 15. Manual Evolution Note

이 문서는 초기 초안이다.  
프로젝트를 진행하면서 아래 기준에 해당하는 개선점이 발견되면 `_feedback.md` 또는 Phase Report에 개선 후보로 기록한다.

Manual에 반영할 만한 경우:

- 같은 문제가 반복될 가능성이 있다.
- 다른 프로젝트에서도 재사용 가능하다.
- Agent의 범위 이탈, 데이터 손상, 실행 오류, 문서 불일치를 방지한다.
- 사용자 확인이 필요한 기준을 더 명확히 만들 수 있다.
- Planning 결과가 너무 길거나 너무 추상적으로 작성되는 문제를 줄일 수 있다.

Manual에 반영하지 않아도 되는 경우:

- 이번 프로젝트에만 해당하는 일회성 문제
- 단순 오타
- 코드 주석으로 충분한 내용
- 너무 세부적인 개인 취향
- 한 번만 발생한 경미한 오류

Manual은 완성된 규칙집이 아니라, 실제 작업 경험을 통해 점진적으로 개선되는 운영 자산이다.
