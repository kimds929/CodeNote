# Planning Manual (`_plan.md`)

## 1. Purpose

이 문서는 Agent가 구현을 시작하기 전에 **작업 목표, 범위, 실행 순서, 검증 기준**을 짧고 명확하게 정리하기 위한 Planning 매뉴얼이다.

Planning의 목적은 다음 4가지다.

1. 이번 작업에서 무엇을 할지 정한다.
2. 이번 작업에서 무엇을 하지 않을지 정한다.
3. 어떤 파일을 만들거나 수정할지 정한다.
4. 완료 여부를 어떻게 검증할지 정한다.

---

## 2. Workflow

Agent는 기본적으로 아래 흐름을 따른다.

```text
Planning → Execution → Validation → Improvement → Feedback
```

Planning 단계에서는 구현을 시작하지 않는다.  
단, 아래 확인 작업은 허용한다.

- 프로젝트 문서 확인
- 기존 코드와 폴더 구조 확인
- 데이터 파일 목록과 스키마 확인
- 실행 환경과 라이브러리 확인
- 이전 작업 보고서 확인

---

## 3. When Planning is Required

아래 경우에는 Planning을 먼저 수행한다.

- 새로운 Phase 또는 주요 기능을 시작할 때
- 파일 구조, 데이터 구조, 설계 방향이 바뀔 수 있을 때
- 새로운 데이터, 라이브러리, API를 사용할 때
- 기존 산출물을 수정해야 할 때
- 사용자 확인이 필요한 선택지가 있을 때
- 에러 수정이 현재 범위를 넘을 가능성이 있을 때

단순 오타 수정, 짧은 문서 보완, 작은 스타일 수정은 간단한 계획 요약만 작성하고 진행할 수 있다.

---

## 4. Required Context

Planning 전에 가능한 범위에서 아래 자료를 확인한다.

### Project Manuals

- `AGENTS.md`
- `_manuals/project_requirements.md`
- `_manuals/Workspace_Structure.md`
- `_manuals/envs.md`
- `_manuals/data.md`
- `_manuals/_execute.md`
- `_manuals/_validate.md`
- `_manuals/_feedback.md`

### Previous Outputs

- `_reports/` 내 최신 보고서
- 이전 Phase의 plan / validation / improvement 문서
- 이전 Phase에서 생성된 코드, 문서, 설정 파일

### Actual Workspace

- 현재 폴더 구조
- `data/` 내 실제 파일 목록
- 현재 사용 가능한 코드와 문서
- 실행 환경과 설치된 라이브러리

확인할 수 없는 내용은 임의로 가정하지 않고 `Risks / Open Questions`에 기록한다.

---

## 5. Planning Principles

### 5.1. Goal First

현재 작업의 목표를 결과 중심으로 1~3줄로 작성한다.

나쁜 예:

```text
RAG 만들기
대시보드 구현
```

좋은 예:

```text
`data/claim.csv`를 로드해 제품별 클레임 건수를 집계하고,
Streamlit 대시보드 첫 화면에 표와 요약 지표로 표시한다.
```

### 5.2. Scope Control

현재 작업의 `In-Scope`와 `Out-of-Scope`를 반드시 구분한다.

- 현재 Phase에서 할 일만 계획한다.
- 다음 Phase 작업은 구현하지 않는다.
- 범위를 넘는 변경은 사용자 확인 후 진행한다.

### 5.3. Small Executable Tasks

계획은 실행 가능한 단위로 쪼갠다.

- 파일 경로
- 함수 / 모듈 / 컴포넌트명
- 입력 데이터
- 출력 결과
- 완료 판단 기준

위 항목이 드러나도록 작성한다.

### 5.4. Check Dependencies

아래 항목을 먼저 확인한다.

- 필요한 데이터
- 필요한 기존 코드
- 필요한 라이브러리
- 외부 API 또는 도구
- 참조해야 할 매뉴얼
- 접근하면 안 되는 민감 정보

`.env`, API Key, 계정 정보 등은 열람하거나 노출하지 않는다.

### 5.5. Validation Attached

모든 계획에는 최소 검증 기준을 함께 작성한다.

- 실행되는가?
- 기대 출력이 생성되는가?
- 실제 입력 데이터를 사용하는가?
- 현재 범위를 넘지 않았는가?
- 문서와 구현이 일치하는가?

### 5.6. Minimal First

처음부터 거대한 구조를 만들지 않는다.

```text
작게 동작하는 구조 → 검증 → 개선 → 확장
```

LangGraph, Multi-Agent, AgentExecutor 같은 복잡한 구조는 MVP가 동작한 뒤 도입한다.

---

## 6. Planning Procedure

Agent는 아래 순서로 계획을 작성한다.

1. **Work Unit 정의**
   - Type: Project / Phase / Feature / Bugfix / Refactoring
   - Name
   - Related Phase
   - Current Status

2. **Context 확인**
   - 참고한 문서
   - 확인한 코드와 폴더 구조
   - 확인한 데이터
   - 이전 산출물

3. **Goal 정리**
   - 이번 작업의 핵심 목표 1~3줄

4. **Scope 정의**
   - In-Scope
   - Out-of-Scope

5. **Dependencies / Constraints 점검**
   - 필요한 입력
   - 필요한 환경
   - 필요한 도구
   - 확인되지 않은 정보
   - 접근 금지 정보

6. **Target Outputs 정의**
   - To Create
   - To Modify
   - To Review / Validate

7. **Task Breakdown 작성**
   - 실행 순서대로 체크리스트 작성
   - 가능한 한 파일명, 함수명, 출력물을 구체화

8. **Validation Plan 작성**
   - 검증 항목
   - 테스트 방법
   - 기대 결과
   - 통과 기준

9. **Risks / Open Questions 정리**
   - 리스크
   - 불확실한 점
   - 사용자 확인 필요사항

10. **Execution Handoff 작성**
   - 실행 전에 계획 요약을 남기고 `_execute.md` 기준으로 실행 단계로 이동한다.

---

## 7. Plan Document Template
작업 규모에 따라 아래 둘 중 하나를 사용한다.

### A. Quick Plan
작은 작업에 사용한다.
```markdown
# Execution Handoff Summary

## Current Work
- 작업 단위:
- 관련 Phase:

## Confirmed Scope
- 이번에 수행할 범위:

## Files to Create / Modify
- 생성:
- 수정:

## Validation Plan
- 핵심 검증 기준:

## User Confirmation Needed
- [ ] 확인 필요사항

## Next Action
- `_manuals/_execute.md` 기준으로 실행한다.
```

### B. Full Plan
Phase, 주요 기능, 구조 변경 작업에 사용한다.

```markdown
# [Work Unit Name] Plan

## 1. Work Unit
- Type:
- Name:
- Related Phase:
- Current Status:

## 2. Goal
- 이번 작업의 핵심 목표:

## 3. Context Summary
- 참고한 문서:
- 확인한 파일 / 폴더:
- 확인한 데이터:
- 이전 산출물:

## 4. Scope

### In-Scope
- [ ] 

### Out-of-Scope
- [ ] 

## 5. Dependencies / Constraints
- 필요한 입력:
- 필요한 기존 코드:
- 필요한 환경 / 라이브러리:
- 확인되지 않은 정보:
- 접근 금지 정보:

## 6. Target Outputs

### To Create
- `path/file`: 역할

### To Modify
- `path/file`: 수정 목적

### To Review / Validate
- `path/file`: 검토 목적

## 7. Task Checklist
- [ ] Step 1.
- [ ] Step 2.
- [ ] Step 3.

## 8. Validation Plan
- 검증 항목:
- 테스트 방법:
- 기대 결과:
- 최소 통과 기준:

## 9. Risks / Open Questions
### Risks
- [ ] 

### Open Questions
- [ ] 

## 10. Exit Criteria
- [ ] 목표가 충족되었는가
- [ ] 최소 검증 기준을 통과했는가
- [ ] 현재 범위를 벗어나지 않았는가
- [ ] 문서와 구현이 일치하는가
```

---

## 8. Prohibited Planning Patterns

아래 방식은 금지한다.

### 8.1. 너무 추상적인 계획

```text
기능 구현
RAG 만들기
대시보드 만들기
```

### 8.2. 범위 침범

```text
현재 Phase가 데이터 로더 구현인데 Chatbot UI까지 함께 만든다.
```

### 8.3. 의존성 무시

```text
data/ 폴더 확인 없이 데이터 구조를 가정한다.
```

### 8.4. 검증 없는 계획

```text
구현 단계만 있고 실행 테스트나 완료 기준이 없다.
```

### 8.5. 사용자 확인 없는 임의 결정

```text
데이터 구조 변경, 라이브러리 추가, API 비용 발생 작업을 임의로 진행한다.
```

---

## 9. Manual Improvement

Planning 중 반복되는 문제나 재사용 가능한 개선점이 발견되면 `_feedback.md` 또는 Phase Report에 기록한다.

기록할 만한 항목:

- 범위 이탈을 막는 규칙
- 반복되는 실행 오류
- 데이터 손상을 막는 기준
- 사용자 확인이 필요한 상황
- 다른 프로젝트에도 재사용 가능한 규칙

단순 오타, 일회성 취향, 이번 프로젝트에만 해당하는 세부사항은 매뉴얼에 반영하지 않는다.
