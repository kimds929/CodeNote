# Planning Manual (`_plan.md`)

## 1. Purpose
이 문서는 Agent가 작업을 시작하기 전, **수행 방향성을 확립하고 작업 계획 및 검증 계획을 수립할 때 준수해야 하는 계획 수립 메뉴얼**이다.
Agent는 코드를 작성하거나 파일을 수정하기 전에, 현재 작업의 목표와 범위, 대상 파일, 실행 순서, 검증 기준, 리스크에 대한 계획을 수립하고 문서화 해야한다.

이 문서의 목적은 다음과 같다.
- 현재 Phase의 목표와 Scope(작업 범위)를 명확히 한다.
- 추상적인 목표를 실행 가능한 구체적 Task 단위로 분해한다.
- 생성 및 수정 대상 파일을 사전에 정의하여 구조적 충돌을 방지한다.
- 작업 결과를 검증 할 수 있는 계획을 수립하여 결과물의 품질을 담보한다.
- 불명확한 요구사항과 리스크를 사전에 식별하여 `User`와 조율한다.

---

## 2. Agent Workflow

Agent는 기본적으로 아래 흐름을 따른다.

```text
Planning → Execution → Validation → Improvement → Feedback
```

Planning 단계에서는 본격적인 구현을 시작하지 않는다.  
다만 아래 작업은 허용된다.

- workspace 구조 확인
- 관련 문서 확인
- 데이터 파일 목록 및 스키마 확인
- 실행 환경 및 라이브러리 확인
- 이전 보고서 및 기존 코드 확인

Planning의 핵심 질문은 다음과 같다.

```text
1. 지금 무엇을 해야 하는가?
2. 어디까지 해야 하는가?
3. 무엇은 하지 말아야 하는가?
4. 어떤 파일을 만들거나 수정해야 하는가?
5. 어떻게 검증할 것인가?
6. 사용자 확인이 필요한 것은 무엇인가?
```

---

## 3. When Planning is Required
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

단순 오타 수정, 주석 보완, 작은 문서 수정은 간단한 계획 요약만으로 진행할 수 있다.

---

## 4. Required Inputs

Planning 전에 가능한 범위에서 아래 자료를 확인한다.

### 4.1. Project Documents
- `AGENTS.md`
- `_manuals/project_requirements.md`
- `_manuals/Workspace_Structure.md`
- `_manuals/envs.md`
- `_manuals/data.md`
- `_manuals/_execute.md`
- `_manuals/_validate.md`
- `_manuals/_feedback.md`

### 4.2. Previous Outputs
- `_reports/` 내 최신 `.md` 보고서 (존재 시)
- `_manuals/workflow/Phase{i}_plan.md` (존재 시)
- `_manuals/workflow/Phase{i}_Validate.md` (존재 시)
- `_manuals/workflow/Phase{i}_Improve.md` (존재 시)
- `_manuals/workflow/Project_Validate.md` (존재 시)
- 이전 Phase에서 생성된 코드, 문서, 설정 파일

### 4.3. Actual Workspace
- 현재 `Workspace` 폴더 구조
- `data/` 폴더 내 실제 파일 목록
- 현재 Phase에서 사용 가능한 기존 코드 및 문서
- Code 실행 환경 및 Library 정보 (`_manuals/envs.md`)

자료가 없거나 확인할 수 없는 경우 임의로 가정하지 말고 `Risks / Open Questions`에 기록한다.

---

## 5. Core Principles

### 5.1. Goal Clarity
- 현재 작업의 목표를 결과 중심으로 1~3줄로 작성한다.
- 목표가 모호하면 구현 전에 사용자에게 확인한다.

### 5.2. Scope Control
- 현재 Phase 또는 Task 범위에 집중한다.
- 반드시 `In-Scope`와 `Out-of-Scope`를 구분한다.
- 다음 Phase의 구현은 현재 계획에 포함하지 않는다.
- 현재 범위를 넘어서는 구현이 필요하면 사용자 확인을 받는다.

### 5.3. Actionable Breakdown
- 계획은 실제 실행 가능한 단위로 쪼갠다.
- 가능하면 파일, 함수, 모듈, UI 컴포넌트, 테스트 단위로 작성한다.
- 각 작업은 완료 여부를 판단할 수 있어야 한다.
- "A 파일을 읽어 B 함수로 전처리한 뒤 C 파일에 저장한다" 수준으로 구체화하여 Checklist를 작성한다.
- Task는 '데이터 로드, 'UI 구성'처럼 추상적으로 적지 않고 '`tools/loaders.py`에 CSV 로드 함수 구현'과 같이 구체적으로 작성한다.

### 5.4. Dependency Check
현재 작업에 필요한 아래 항목을 먼저 확인한다.

- 데이터 파일
- 기존 코드와 문서
- 실행 환경과 라이브러리
- 외부 API 또는 도구
- 이전 Phase 산출물
- 참조해야 할 manual

`.env`, 인증키, 계정 정보 등 민감한 파일은 열람하거나 노출하지 않는다.

### 5.5. Validation Readiness
- 모든 계획은 검증 가능해야 한다.
- 각 핵심 작업은 검증 항목과 연결되어야 한다.
- Planning 단계에서 최소한의 Validation Plan을 함께 작성한다.
- 가능하다면 정성적 평가기준보다는 정량적인 기준을 바탕으로 검증계획을 수립하고 결과를 검증한다.

### 5.6. User Confirmation First
아래 경우에는 임의로 결정하지 않는다.

- 요구사항이 모호한 경우
- 구현 방식이 2개 이상인 경우
- 데이터 구조 변경이 필요한 경우
- 외부 의존성 추가가 필요한 경우
- 기존 구조를 크게 바꿔야 하는 경우
- 비용이 발생할 수 있는 API 호출이 필요한 경우

### 5.7. No Fake Assumptions
- 존재하지 않는 데이터, 파일, API, 환경을 전제로 계획하지 않는다.
- Mock Data, Dummy Data, Placeholder를 핵심 계획의 전제로 삼지 않는다.
- 테스트 데이터가 필요한 경우에도 프로젝트 규칙 또는 사용자 승인을 따른다.

### 5.8. Minimal First
처음부터 완성형 구조를 계획하지 않고, 반드시 아래 방식에 의거하여 계획을 세운다.
```text
작게 동작하는 구조 → 검증 → 개선 → 확장
```
LangGraph, Multi-Agent, AgentExecutor 같은 고급 구조는 MVP가 정상 동작한 뒤 도입한다.

---

## 6. Planning Procedure

Agent는 작업 시작 시 아래 순서로 Planning을 수행한다.

### Step 1. Current Work 정의
- 현재 작업 단위: Project / Phase / Task / Feature / Bugfix / Refactoring 등
- 현재 작업 이름
- 관련 Phase
- 전체 Workflow에서의 위치

### Step 2. Context 확인
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
- 이번 작업의 목표를 1~3줄로 작성한다.
- 여러 목표가 있다면 우선순위를 구분한다.

### Step 4. Scope 정의

### In-Scope
- 이번 작업에서 수행할 항목

### Out-of-Scope
- 이번 작업에서 수행하지 않을 항목
- 다음 Phase에서 수행할 항목
- 사용자 승인 전에는 수행하면 안 되는 항목


### Step 5. Dependency / Constraint 점검
- 필요한 데이터
- 필요한 기존 산출물
- 필요한 환경 / 라이브러리
- 외부 API 또는 서비스
- 접근하면 안 되는 파일 / 정보
- 아직 확인되지 않은 정보


### Step 6. Target Outputs 도출
이번 작업의 결과로 생성, 수정, 검토되어야 하는 대상을 정의한다.
 Target Outputs는 단순히 “무언가 만든다”가 아니라, 이번 작업 이후 실제로 어떤 파일, 문서, 코드, 설정, 산출물이 생성되거나 변경되어야 하는지를 명확히 하는 단계다.

- To Create
  - 새로 생성할 파일 또는 산출물

- To Modify
  - 수정할 파일 또는 산출물

- To Review / Validate
  - 검토하거나 검증할 파일 또는 산출물


각 항목에는 파일 경로와 역할을 함께 작성한다.

### Step 7. Task Breakdown
- 논리적 순서대로 checklist 형태로 작성한다.
- 코드 작업은 파일명, 함수명, 모듈명을 구체적으로 작성한다.
- 데이터 작업은 입력 파일, 처리 방식, 출력 결과를 명시한다.
- UI 작업은 표시 위치, 입력 방식, 출력 화면을 명시한다.

### Step 8. Validation Planning
- 검증 목표
- 검증 항목
- 테스트 시나리오
- 기대 결과
- 최소 통과 기준
- 실패 시 처리 방식

### Step 9. Risks / Open Questions 정리
- 리스크
- 불확실성
- 사용자 확인 필요사항
- 현재 범위 밖의 문제

### Step 10. Planning 문서화
권장 경로:

```text
_manuals/workflow/Phase{i}_plan.md
_manuals/workflow/Phase{i}_Validate.md
```

프로젝트에서 다른 naming convention을 지정한 경우 해당 규칙을 따른다.

---

## 7. Required Outputs

Planning 결과로 아래 산출물을 만든다.
- Plan Document
- Validation Plan Document

Plan Document에는 아래 항목이 포함되어야 한다.

- 현재 작업 단위
- 목표
- Context Summary
- In-Scope / Out-of-Scope
- 의존성 / 제약사항
- 생성 / 수정 / 검토 대상
- 상세 작업 단계
- 검증 계획 링크
- 리스크 / 질문사항
- 완료 기준

Validation Plan에는 아래 항목이 포함되어야 한다.

- 검증 목표
- 검증 항목
- 테스트 시나리오
- 기대 결과
- 최소 통과 기준
- 실패 처리 방식

---

## 8. Plan Document Template

```markdown
# [Work Unit Name] Plan

## 1. Work Unit
- Type:
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
- [ ] 수행 항목 1
- [ ] 수행 항목 2

### Out-of-Scope
- [ ] 현재 범위 밖 항목
- [ ] 다음 Phase 항목
- [ ] 사용자 승인 전 진행 금지 항목

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
  - 세부 작업

- [ ] Step 2. [작업명]
  - 세부 작업

## 8. Validation Link
- 관련 검증 문서: `path/or/name`

## 9. Risks / Open Questions
### Risks
- [ ] 리스크 1

### Open Questions
- [ ] 사용자 확인 필요사항 1

## 10. Exit Criteria
- [ ] 핵심 목표가 충족되었는가
- [ ] 최소 검증 기준을 만족하는가
- [ ] 문서와 구현이 일관되는가
- [ ] 현재 범위를 벗어난 구현이 포함되지 않았는가
```

---

## 9. Validation Plan Template

```markdown
# [Work Unit Name] Validation Plan

## 1. Validation Goal
- 이번 작업 결과물이 충족해야 하는 핵심 검증 목표

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
- 실패 원인을 기능 / 데이터 / 환경 / 설계 / 범위 문제로 분류한다.
- 현재 범위 내 수정 가능 여부를 판단한다.
- 현재 범위를 벗어나면 사용자에게 보고하고 선택지를 제시한다.
```

---

## 10. Planning Quality Checklist

Planning 문서를 작성한 뒤 아래 항목을 자체 점검한다.

- [ ] 현재 작업 단위가 명확한가
- [ ] Goal이 결과 중심으로 작성되었는가
- [ ] In-Scope / Out-of-Scope가 분리되었는가
- [ ] 의존성과 제약사항이 기록되었는가
- [ ] 생성 / 수정 / 검토 대상이 구분되었는가
- [ ] 작업이 실행 가능한 단위로 분해되었는가
- [ ] 검증 계획이 정의되었는가
- [ ] 사용자 확인 필요사항이 분리되었는가
- [ ] 확인되지 않은 가정을 사실처럼 쓰지 않았는가
- [ ] Mock Data를 전제로 하지 않았는가
- [ ] 현재 범위를 벗어난 항목이 섞이지 않았는가
- [ ] 보안상 열람하면 안 되는 파일을 참조하지 않았는가

---

## 11. Prohibited Planning Patterns

아래 Planning은 금지한다.

### 11.1. 지나치게 추상적인 계획
금지 예:

```text
기능 구현
서비스 개발
RAG 만들기
대시보드 구현
```

### 11.2. 범위 침범 계획
금지 예:

```text
현재 Phase가 VectorDB 구성인데 Chatbot UI까지 함께 구현한다.
승인되지 않은 기능을 추가한다.
```

### 11.3. 의존성 무시 계획
금지 예:

```text
data/ 폴더 확인 없이 데이터 로더를 설계한다.
환경 문서 확인 없이 라이브러리를 추가한다.
```

### 11.4. 검증 없는 계획
금지 예:

```text
구현 작업만 있고 실행 테스트 또는 완료 기준이 없다.
```

### 11.5. 사용자 확인 없는 임의 결정
금지 예:

```text
요구사항이 애매한데 Agent가 독단적으로 설계를 확정한다.
```

---

## 12. Execution Handoff

Planning이 완료되면 Execution 단계로 넘어가기 전에 아래 내용을 사용자에게 제시한다.

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
- [ ] 확인 필요사항

## Next Action
- 사용자 확인 후 `_manuals/_execute.md` 기준으로 실행을 시작한다.
```

아래 경우에는 반드시 사용자 확인을 받는다.

- 현재 Phase 범위를 넘어서는 구현
- 외부 라이브러리 추가
- 데이터 구조 변경
- 저장 위치 변경
- 기존 설계 방향 변경
- 보안 또는 민감 정보와 관련된 작업
- 비용이 발생할 수 있는 API 호출

---

## 13. Manual Evolution Note
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
