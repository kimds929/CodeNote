## 핵심 답

`Generator of VibeCoding Supervisor`의 workflow는 **프로젝트를 직접 개발하는 절차가 아니라, 그 프로젝트를 감독할 Supervisor 운영체계를 설계·생성·검증하는 절차**입니다.

즉 다음 두 workflow를 구분해야 합니다.

```text
① Generator Workflow
   프로젝트 분석 → Supervisor 설계 → Supervisor Package 생성

② Generated Supervisor Runtime Workflow
   실제 프로젝트 관찰 → 다음 작업 결정 → Coding Agent 지시 → 검증 → 상태 갱신
```

현재 질문은 ①에 해당합니다.  
이 Generator의 역할은 “이 프로젝트에서는 어떤 workflow, 어떤 통제 수준, 어떤 평가 기준, 어떤 evidence, 어떤 handoff 방식이 필요한가?”를 결정해 프로젝트별 Supervisor Package를 만드는 것입니다.

---

# 전체 순서

```text
0. Invocation / Scope Confirmation
→ 1. Environment & Capability Discovery
→ 2. Project Baseline & Repository Discovery
→ 3. Adaptive Human Interview
→ 4. Project Classification & Risk Assessment
→ 5. Information Sufficiency Decision
→ 6. Research Planning and Evidence Synthesis
→ 7. Supervisor Architecture Composition
→ 8. Goal / Evaluation / Authority Contract Design
→ 9. Workflow and Execution-Control Composition
→ 10. Supervisor Package Generation
→ 11. Package Validation / Dry Run
→ 12. Human Review and Finalization
```

중요한 원칙은 **항상 모든 단계를 같은 깊이로 수행하지 않는 것**입니다.

```text
단순·저위험 프로젝트
→ 짧은 발견 → 간단한 인터뷰 → 최소 package 생성

고위험·불확실·기존 시스템 프로젝트
→ 깊은 Repo 조사 → 추가 인터뷰 → Research → 세분화된 workflow/control 설계
```

---

# 0. Invocation / Scope Confirmation

## 목적

Generator가 무엇을 생성해야 하는지와, Generator 자신이 어디까지 할 수 있는지를 먼저 정합니다.

## 확인할 것

```text
- 대상 프로젝트 Root
- 생성 대상: 새 프로젝트 / 기존 프로젝트 / 특정 기능 또는 문제
- Human의 최상위 목표
- 현재 사용하는 Agent 또는 작업 환경
- Supervisor Package를 둘 위치
- 생성 중 파일 작성 허용 여부
- 외부 Research 가능 여부
- 민감정보·개인정보·사내정보 제약
```

## 산출물

```text
Generation Request
Capability Profile
Initial Privacy / Authority Constraints
```

여기서 Capability Profile은 특정 Agent 이름보다 다음 능력을 기록합니다.

```text
- Repository 읽기 가능 여부
- Git 상태 확인 가능 여부
- Terminal / Test 실행 가능 여부
- 파일 작성 가능 여부
- Web Research 가능 여부
- Subagent 호출 가능 여부
- Human이 Evidence를 수동 전달해야 하는지
```

이 단계가 있어야 범용 Agent workflow로 동작할 수 있습니다.

---

# 1. Environment & Capability Discovery

## 목적

“이 프로젝트에서 Supervisor가 실제로 관찰하고 실행할 수 있는 범위”를 파악합니다.

## 확인할 것

```text
- Repo 접근 가능 여부
- Git 사용 여부
- test / lint / typecheck / build 환경
- CI/CD 존재 여부
- Browser / Runtime / Visual 검증 가능 여부
- MCP 또는 외부 도구 존재 여부
- 프로젝트 문서 위치
- Agent가 artifact를 쓸 수 있는 영역
```

## 핵심 판단

```text
Evidence를 자동 수집할 수 있는가?
아니면 Human-mediated 방식으로 받아야 하는가?
```

예:

```text
Repo/Git/Test 직접 접근 가능
→ Supervisor가 실제 Evidence를 직접 확인

접근 불가
→ Coding Agent 또는 Human에게 정해진 Evidence Bundle 요청
```

## 산출물

```text
Capability Profile
Evidence Collection Strategy
Human-Mediation Requirements
```

---

# 2. Project Baseline & Repository Discovery

## 목적

프로젝트의 현재 사실을 추정하지 않고 확인합니다.  
특히 기존 프로젝트에서는 이후 모든 판단의 기준점이 됩니다.

## 수행 내용

```text
- 실제 Project Root 확인
- Branch 확인
- HEAD 확인
- Working Tree 상태 확인
- 미커밋 변경 확인
- 기존 문서, 설정, 테스트, CI, deployment 구조 탐색
- 주요 코드 경계와 dependency 파악
- 이미 존재하는 정책, ADR, issue, roadmap 확인
```

## 기존 미커밋 변경 처리

확정한 정책에 따라:

```text
기존 변경 발견
→ 파일 목록 / 알려진 목적 / 현재 작업 영향 기록

현재 작업과 명확히 분리 가능
→ 보호 범위로 지정

출처·목적·영향 불명확
→ HOLD 또는 Human 확인
```

## 산출물

```text
Project Baseline
Repository Discovery Summary
Existing Constraints
Initial Risks
```

---

# 3. Adaptive Human Interview

## 목적

Repo만 보고 알 수 없는 의도, 우선순위, 수용 기준을 Human에게서 얻습니다.

모든 질문을 한 번에 던지지 않습니다. 앞 단계에서 확인된 정보와 불확실성에 따라 필요한 질문만 합니다.

## 공통 질문 범주

```text
- 프로젝트의 Goal은 무엇인가?
- 무엇이 완료 또는 성공인가?
- 무엇이 절대 실패하면 안 되는가?
- 기능, 품질, 속도, 비용, 안전성 중 무엇이 우선인가?
- 일정·예산·배포·보안 제약은 무엇인가?
- Human이 반드시 승인해야 하는 결정은 무엇인가?
- 현재 사용 중인 Agent 또는 운영 방식은 무엇인가?
```

## 상황별 추가 질문

| 프로젝트 신호 | 추가 인터뷰 |
|---|---|
| 기존 코드가 복잡함 | 변경 금지 영역, 기존 장애 이력, 호환성 요구 |
| DB 관련 | 데이터 손실 허용 여부, rollback, migration 승인자 |
| 보안 관련 | 위협 모델, credential 처리, 접근 권한, 감사 요구 |
| UI 관련 | visual acceptance 기준, 지원 브라우저, 디자인 기준 |
| RAG/AI 관련 | 평가셋, 비용 한도, 정답성 기준, hallucination 허용 범위 |
| 배포 관련 | 환경, rollback, release 승인, 운영 영향 |
| 요구사항이 모호함 | 우선순위, 제외 범위, 판단 권한 |

## 산출물

```text
Human Intent Record
Success Definition
Acceptance Criteria
Approval Boundaries
```

---

# 4. Project Classification & Risk Assessment

## 목적

프로젝트에 맞는 Supervisor 운영 방식과 workflow를 고르기 위한 분류를 수행합니다.

## 분류 기준

```text
Task Type
- New Feature
- Bug Fix
- Refactor
- Migration
- Research
- Prototype
- Performance
- Security
- Data / ML
- AI / RAG / Agent
- Infrastructure / Deployment

Project Context
- Greenfield / Existing / Legacy
- Project Stage
- Tech Stack / Environment

Control Factors
- Risk
- Complexity
- Uncertainty
- Reversibility
- Impact Surface
- Verification Feasibility
```

## 예시

```text
버그 수정 + 높은 불확실성
→ 재현과 원인 분석 중심 workflow

UI prototype + 저위험
→ 빠른 prototype과 visual verification 중심 workflow

DB migration + 낮은 가역성
→ 조사·승인·구현·검증·checkpoint 분리

새 AI Framework 도입
→ Research, prototype, evaluation contract 강화
```

## 산출물

```text
Project Profile
Risk Profile
Workflow Selection Factors
Initial Control Level
```

---

# 5. Information Sufficiency Decision

## 목적

정보가 충분한지 판단하고, 부족하면 무작정 package를 생성하지 않습니다.

## 충분성 기준

최소한 아래가 명확해야 합니다.

```text
- 프로젝트 Goal
- 성공 또는 수용 기준
- 현재 상태 또는 Baseline
- 작업 범위와 핵심 제약
- 주요 위험
- 검증 가능한 Evidence 방식
- Human 승인 경계
- 사용 가능한 Agent/환경 capability
```

## 분기

```text
충분
→ Research 또는 Supervisor 설계로 진행

보완 가능
→ 추가 Adaptive Interview

핵심 정보 부재
→ HOLD
```

예를 들어 “로그인 기능 개선”이라는 요구만 있고, 기존 인증 구조·허용 변경 범위·보안 정책이 전혀 없다면 구현 workflow를 생성하면 안 됩니다.

---

# 6. Research Planning & Evidence Synthesis

## 목적

Research가 필요할 때만, 최신 외부 지식과 프로젝트 상황을 연결합니다.

## Research가 필요한 대표 상황

```text
- 새로운 Framework / Library / Tool
- Architecture Decision
- Security / DB / Deployment 관련 고위험 변경
- 반복 실패
- 요구사항 모호성
- 기존 계획과 실제 결과의 큰 불일치
- 내부 지식 또는 기존 practice의 최신성 불확실
```

## 수행 순서

```text
Research Question 정의
→ 외부 전달 가능 정보 검토
→ Search
→ 출처 분류
→ Evidence 평가
→ Project Fit 평가
→ 채택 / 기각 / 보류
→ Research Ledger 기록
```

## 개인정보·민감정보 정책

확정한 정책에 따라 외부 Research에는 기본적으로 아래만 사용합니다.

```text
- 기술명
- 버전
- 일반화된 문제 설명
- 공개 가능한 오류 유형
```

다음은 자동 제외입니다.

```text
- 소스 코드 원문
- Secret
- 개인정보
- 내부 URL
- 고객·사내 식별자
- 민감 로그·설정값
```

## 산출물

```text
Research Plan
Research Ledger
Adopted Practices
Rejected Alternatives
Confidence / Freshness Record
```

---

# 7. Supervisor Architecture Composition

## 목적

앞에서 수집한 정보를 바탕으로 해당 프로젝트용 Supervisor의 구성요소를 선택합니다.

여기서 Generator는 “모든 기능을 구현”하는 것이 아니라, 필요한 capability를 조합합니다.

## 선택 가능한 capability 예시

```text
- Project Discovery
- Research
- Execution Control
- Evidence Verification
- Evaluation
- Handoff
- Planning
- Future Wiki Integration
- Future Learning Telemetry
```

## 예시 조합

### 단순 UI 개선

```text
Discovery
+ Basic Execution Control
+ Visual Verification
+ Lightweight Handoff
```

### Legacy bug fix

```text
Repository Discovery
+ Reproduction Workflow
+ Strong Baseline
+ Evidence Verification
+ Recovery Protocol
+ Regression Evaluation
```

### DB migration

```text
Deep Discovery
+ Research
+ High-Risk Approval
+ Split Controlled Work Units
+ Rollback Requirement
+ Checkpoint Policy
+ Human Final Approval
```

### AI/RAG 실험

```text
Research
+ Hypothesis / Prototype Workflow
+ Evaluation Contract
+ Cost / Quality Measurement
+ Experiment Telemetry
```

## 산출물

```text
Supervisor Architecture Plan
Selected Capability Set
Invocation Strategy
```

여기서 Invocation Strategy는 다음처럼 설계합니다.

```text
전문 기능 필요
→ 현재 환경이 Skill/Subagent 호출 지원
   → 독립 Specialist Skill 또는 Subagent 호출

→ 호출 지원 없음
   → 동일 Supervisor가 해당 protocol 실행

→ 필요한 Evidence 접근 불가
   → Human-mediated 요청
```

---

# 8. Goal / Evaluation / Authority Contract Design

## 목적

프로젝트를 실제로 시작하기 전에 “무엇이 성공이며, 누가 무엇을 승인하는가”를 명시합니다.

## 생성할 계약

### Goal Contract

```text
- Project Goal
- Scope
- Non-goals
- Critical Requirements
- Human Acceptance Criteria
- Constraints
- Terminal Conditions
```

### Evaluation Contract

```text
- Hard Gates
- Quality Dimensions
- Measurement Method
- Evidence Source
- Evaluation Timing
- Evaluator Authority
- PASS / FAIL / HOLD 기준
```

### Authority Contract

```text
Supervisor 자동 허용:
- 상태, telemetry, research ledger, handoff 업데이트
- 저·중위험 작업의 Evidence 기반 판정

Human 승인 필요:
- Architecture
- Security
- DB migration
- Deployment
- 범위 확대
- 실제 commit
- Push
- Deployment
```

## 산출물

```text
Goal Contract
Evaluation Plan
Authority Policy
Privacy Policy
```

---

# 9. Workflow & Execution-Control Composition

## 목적

“어떻게 문제를 풀 것인가”와 “어떻게 안전하게 실행·판정할 것인가”를 함께 생성합니다.

이 둘은 다릅니다.

```text
Workflow
= 문제 해결의 방법과 순서

Controlled Work Unit
= 실제 변경 작업을 Baseline·범위·Evidence·판정으로 통제하는 단위
```

## Workflow 조립 예시

```text
Bug Fix:
Reproduce
→ Explore
→ Root Cause
→ Minimal Fix
→ Regression Verification

Feature:
Specify
→ Explore
→ Plan
→ Implement
→ Verify
→ Review

Research / Prototype:
Hypothesis
→ Research
→ Prototype
→ Evaluate
→ Decide
```

## Controlled Work Unit 조립

각 작업 단위에는 다음이 포함됩니다.

```text
- 목적
- Baseline
- 진입 조건
- 허용 범위
- 보호 범위
- 실행 절차
- Evidence 요구
- 검증 기준
- PASS / FAIL / HOLD 기준
- Recovery 조건
- Checkpoint 방침
```

### 적응형 분리 원칙

```text
저위험·가역적 변경
→ 구현 + 기본 검증을 하나의 Controlled Work Unit으로 결합

고위험·비가역·불확실·광범위 변경
→ 조사 → 구현 → 검증 → checkpoint를 분리
```

## 산출물

```text
Project-Specific Workflow
Controlled Work Unit Templates
Baseline / Evidence / Recovery Policy
```

---

# 10. Supervisor Package Generation

## 목적

이제 설계된 내용을 실제 프로젝트가 사용할 수 있는 package로 생성합니다.

예시 구조:

```text
.supervisor/
├── PROJECT_PROFILE
├── GOAL_CONTRACT
├── EVALUATION_PLAN
├── ROADMAP
├── CURRENT_STATE
├── SUPERVISOR_POLICY
├── capability-profile
├── workflows/
├── execution-control/
├── research/
├── decisions/
├── telemetry/
├── sessions/
└── backlog/
```

중요한 점은 entry instruction이 모든 내용을 담지 않는다는 것입니다.

```text
Entry
→ Current State
→ Active Workflow
→ Relevant Controlled Work Unit
→ 필요한 Decision / Research / Handoff
```

이 progressive disclosure 방식으로 context 오염을 줄입니다.

## 산출물

```text
Project-Specific Supervisor Package
Initial Roadmap
Initial Current State
Initial Controlled Work Unit
Bootstrap Prompt
```

---

# 11. Package Validation / Dry Run

## 목적

생성된 Supervisor가 단순히 문서를 많이 만든 것이 아니라, 실제로 동작 가능한지를 검증합니다.

## 검증 질문

```text
- Goal과 success criteria가 명확한가?
- 현재 상태와 Baseline이 구분되는가?
- workflow가 프로젝트 특성에 맞는가?
- 모든 작업에 과도한 통제를 강제하지 않는가?
- 고위험 변경은 충분히 통제되는가?
- Evidence 요구가 실제 가능한가?
- Human 승인 지점이 명확한가?
- privacy policy가 Research 흐름에 반영되었는가?
- PASS_CANDIDATE와 공식 PASS가 구분되는가?
- FAIL 후 Recovery path가 존재하는가?
- Agent capability가 부족할 때 Human-mediated 대안이 있는가?
- handoff가 전체 대화 복사가 아니라 상태 복구를 지원하는가?
```

## Dry Run

실제 코드를 수정하지 않고 다음 상황을 가상으로 통과시켜 봅니다.

```text
- 단순 변경
- 불확실한 버그
- 고위험 DB 또는 보안 변경
- 테스트 실패 후 Recovery
- 세션 handoff
```

## 산출물

```text
Validation Report
Detected Gaps
Revision List
```

---

# 12. Human Review & Finalization

## 목적

생성된 Supervisor가 Human의 실제 의도와 운영 선호를 반영하는지 최종 확인합니다.

## Human이 검토할 핵심 항목

```text
- Goal과 Non-goal이 맞는가?
- 성공 기준이 현실적인가?
- Human 승인 범위가 적절한가?
- 위험 분류가 맞는가?
- Research privacy 정책이 충분한가?
- workflow가 너무 무겁거나 너무 느슨하지 않은가?
- 체크포인트와 commit 승인 방식이 맞는가?
- 평가 방식이 프로젝트 목적에 맞는가?
```

## 결과 분기

```text
Approved
→ Supervisor Package 활성화

Revision Required
→ 해당 단계로 돌아가 수정

Fundamental Ambiguity
→ 추가 interview 또는 HOLD
```

---

# Generator Workflow를 한 장으로 요약하면

```text
Request
  ↓
Capability Discovery
  ↓
Baseline / Repo Discovery
  ↓
Adaptive Interview
  ↓
Classification / Risk Assessment
  ↓
Information Sufficiency Check
  ├── 부족 → 추가 Interview 또는 HOLD
  ↓
Conditional Research
  ↓
Supervisor Architecture Composition
  ↓
Goal / Evaluation / Authority Contract
  ↓
Workflow + Execution-Control Composition
  ↓
Supervisor Package Generation
  ↓
Dry Run / Validation
  ↓
Human Review
  ↓
Activate Project Supervisor
```

---

# Generator와 생성된 Supervisor의 경계

마지막으로 이 구분이 중요합니다.

| Generator가 하는 일 | 생성된 Supervisor가 하는 일 |
|---|---|
| 프로젝트를 분석하고 감독 체계를 설계 | 실제 개발 진행을 계속 감독 |
| workflow와 정책을 조합 | 현재 상황에 맞는 다음 행동 선택 |
| Goal/Evaluation/Authority 계약 생성 | Coding Agent 작업 지시 생성 |
| capability와 risk에 맞는 통제 수준 선택 | Baseline, Evidence, PASS/FAIL/HOLD 운영 |
| package와 template 생성 | 상태·roadmap·handoff·telemetry 갱신 |
| 생성 결과 검증 | 실패 시 Recovery와 replan 수행 |

즉:

```text
Generator
= 프로젝트별 Supervisor를 설계·조립하는 Compiler

Supervisor
= 실제 프로젝트 진행 중 판단·통제·조정하는 Runtime
```

이 순서로 구성하면 Generator는 너무 많은 기능을 직접 수행하는 거대 Skill이 되지 않고, 프로젝트마다 필요한 감독 능력만 선택·조합하는 상위 workflow로 유지될 수 있습니다.