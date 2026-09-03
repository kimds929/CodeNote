전체 그림부터 보면, 이 Workflow는 **프로젝트를 직접 개발하는 절차가 아니라, “이 프로젝트를 어떻게 감독할 Supervisor를 만들 것인가”를 결정하는 생성 절차**입니다. 즉 앞부분에서는 프로젝트와 환경을 이해하고, 중간에서는 위험·근거·평가기준을 설계하며, 뒷부분에서는 실제 Supervisor Package를 만들고 검증한 뒤 Human 승인으로 활성화합니다. 

전체 흐름은 크게 **① 이해 → ② 판단 → ③ 설계 → ④ 생성 → ⑤ 검증**으로 보면 쉽습니다.

```text
[이해]
0 Scope 확인
→ 1 환경/능력 확인
→ 2 프로젝트 현재상태 확인
→ 3 Human 의도 확인

[판단]
→ 4 프로젝트 유형/위험 분류
→ 5 정보가 충분한지 판단
→ 6 필요 시 Research

[설계]
→ 7 Supervisor 구조 선택
→ 8 Goal/Evaluation/Authority 규칙 확정
→ 9 실제 Workflow와 실행통제 설계

[생성/검증]
→ 10 Package 생성
→ 11 Dry Run으로 검증
→ 12 Human 최종 승인
```

특히 이 Generator는 고정된 Supervisor 하나를 만드는 것이 아니라, 프로젝트의 위험·복잡도·불확실성·Tool capability에 따라 필요한 Supervisor 구조를 달리 만들어야 합니다. 

---

## 0. Invocation / Scope Confirmation

**목적·역할**
이번 Generator 실행에서 **무엇을 대상으로, 어디까지 Supervisor를 만들 것인지 경계부터 확정**합니다. 잘못된 프로젝트나 범위를 대상으로 설계를 시작하는 것을 막는 시작점입니다. 

**왜 필요한가**
Scope가 불명확하면 이후 Repo 조사·질문·Research가 전부 엉뚱한 방향으로 갈 수 있습니다.

**예시**
“전체 쇼핑몰 프로젝트가 아니라 **결제 모듈 개선 작업을 감독할 Supervisor**를 만든다.”

---

## 1. Environment & Capability Discovery

**목적·역할**
Supervisor가 실제로 **무엇을 볼 수 있고, 어떤 Tool을 사용할 수 있는지** 파악합니다. Repo, Git, Test, Web, Subagent 등의 사용 가능 여부에 따라 Supervisor 설계가 달라집니다. 

**왜 필요한가**
사용할 수 없는 기능을 전제로 Supervisor를 만들면 실제 Runtime에서 동작하지 않습니다.

**예시**
“Codex가 Repo·Git·Test에 직접 접근 가능 → 테스트 결과를 Supervisor가 직접 Evidence로 검증.”

---

## 2. Project Baseline & Repository Discovery

**목적·역할**
프로젝트의 **현재 실제 상태(Baseline)** 를 확인합니다. 코드 구조, Branch, 미커밋 변경, 테스트, 기존 정책 등을 이후 판단의 기준점으로 저장합니다. 

**왜 필요한가**
현재 상태를 모르면 어떤 변경이 새로 생겼는지, 무엇이 망가졌는지 정확히 비교할 수 없습니다.

**예시**
“현재 Branch=`feature/login`, 테스트 3개 실패, 인증 관련 미커밋 파일 2개 존재.”

---

## 3. Adaptive Human Interview

**목적·역할**
Repo만으로는 알 수 없는 **Human의 목표·우선순위·성공 기준·금지사항**을 확인합니다. 모든 질문을 고정적으로 묻지 않고 부족한 정보만 추가 질문합니다. 

**왜 필요한가**
기술적으로 맞는 결과라도 Human이 원한 결과와 다르면 프로젝트는 실패이기 때문입니다.

**예시**
“로그인 개편의 최우선 목표는 보안 강화인가?” → “아니요, 기존 사용자 로그인 방식 유지가 최우선.”

---

## 4. Project Classification & Risk Assessment

**목적·역할**
프로젝트를 Feature, Bug Fix, Migration 등으로 분류하고 **Risk·Complexity·Uncertainty·Reversibility**를 평가합니다. 이 결과가 이후 Workflow와 통제 강도를 결정합니다. 

**왜 필요한가**
CSS 수정과 DB Migration에 동일한 Supervisor 절차를 적용하면 한쪽은 과도하고 다른 쪽은 위험합니다.

**예시**
“DB Migration + 낮은 가역성 + 높은 영향도 → 강한 승인·Rollback·Checkpoint 필요.”

---

## 5. Information Sufficiency Decision

**목적·역할**
지금까지 모은 정보만으로 **Supervisor 설계를 시작해도 되는지 Gate 판단**합니다. 부족하면 추가 Interview나 조사를 하고, 핵심 정보가 없으면 HOLD합니다. 

**왜 필요한가**
정보가 부족한 상태에서 설계를 강행하면 잘못된 가정을 Supervisor 정책으로 굳힐 수 있습니다.

**예시**
“인증 개선 요청은 있지만 현재 인증 구조와 보안 제약을 모름 → 설계 진행하지 않고 추가 확인.”

---

## 6. Research Planning and Evidence Synthesis

**목적·역할**
필요한 경우 최신 공식 문서·Best Practice 등을 조사해 **프로젝트에 실제 적용할 Evidence**를 만듭니다. 단순 검색이 아니라 채택/기각 이유까지 기록합니다. 

**왜 필요한가**
Agent 내부 지식만으로는 최신 Framework나 보안·Architecture 판단이 오래되었거나 부정확할 수 있습니다.

**예시**
“새 인증 Library 도입 → 공식 문서와 Maintainer 자료를 비교 → 현재 프로젝트 버전에 맞는 방식만 채택.”

---

## 7. Supervisor Architecture Composition

**목적·역할**
앞에서 얻은 정보를 기반으로 **이 프로젝트 Supervisor에 어떤 기능들을 넣을지 조합**합니다. 모든 프로젝트에 모든 기능을 넣는 것이 아닙니다. 

**왜 필요한가**
Supervisor 자체도 프로젝트에 맞게 가벼워지거나 강해져야 운영비용과 안전성의 균형을 맞출 수 있습니다.

**예시**
“단순 UI 개선 → Basic Execution Control + Visual Verification.”
“Legacy Bug → Reproduction + Evidence Verification + Recovery 강화.”

---

## 8. Goal / Evaluation / Authority Contract Design

**목적·역할**
**무엇을 성공으로 볼지, 어떻게 검증할지, 누가 무엇을 결정할 수 있는지** 명문화합니다. Goal·Evaluation·Authority가 Supervisor의 핵심 운영 계약입니다. 

**왜 필요한가**
성공 기준이나 권한이 없으면 Supervisor가 임의로 PASS를 선언하거나 중요한 결정을 마음대로 바꿀 수 있습니다.

**예시**
“Critical test 100% PASS 필요 / Architecture 변경은 Human 승인 / 상태 파일 수정은 Supervisor 자동 허용.”

---

## 9. Workflow and Execution-Control Composition

**목적·역할**
프로젝트를 **어떤 순서로 해결할지(Workflow)** 와 각 작업을 **어떤 Baseline·Evidence·Gate로 통제할지(Execution Control)** 설계합니다. 

**왜 필요한가**
좋은 문제 해결 절차만으로는 부족하고, 실제 변경이 안전하게 수행되었는지 검증하는 통제 구조도 필요합니다.

**예시**
Bug Fix:
`Reproduce → Root Cause → Minimal Fix → Regression Test`
각 단계마다 Evidence와 PASS/HOLD 조건을 설정.

---

## 10. Supervisor Package Generation

**목적·역할**
앞 단계에서 만든 설계를 실제 Agent가 사용할 수 있는 **파일·정책·Workflow·State Package**로 변환합니다. 

**왜 필요한가**
설계가 대화 속 설명으로만 존재하면 새 Session이나 다른 Agent가 안정적으로 재사용할 수 없습니다.

**예시**

```text
.supervisor/
├── PROJECT_PROFILE
├── GOAL_CONTRACT
├── EVALUATION_PLAN
├── CURRENT_STATE
├── workflows/
├── execution-control/
└── sessions/
```

---

## 11. Package Validation / Dry Run

**목적·역할**
생성된 Supervisor가 **문서상 그럴듯한 것이 아니라 실제 상황에서 동작하는지 가상 실행**합니다. 모순, 과도한 통제, 빠진 Recovery 경로 등을 찾아냅니다. 

**왜 필요한가**
Supervisor 자체가 잘못 설계되어 있으면 이후 Coding Agent를 잘못 통제할 수 있기 때문입니다.

**예시**
“테스트 실패 발생 → Supervisor가 FAIL 판정 → Recovery → 재검증까지 제대로 이어지는지 Dry Run.”

---

## 12. Human Review and Finalization

**목적·역할**
마지막으로 Human이 **목표·위험·승인 범위·Workflow 강도가 실제 의도와 맞는지 확인**하고 최종 활성화합니다. 

**왜 필요한가**
Goal과 최종 Acceptance의 최고 권한은 결국 Human에게 있기 때문입니다. 실제 Evidence보다 Agent 발언을 우선하지 않되, Human의 Goal·Intent·Acceptance 판단은 별도 최상위 권한을 갖도록 설계되어 있습니다. 

**예시**
“Workflow가 지나치게 무겁다” → Human Review에서 수정 요청 → Step 9로 돌아가 간소화 후 재검증.

---

## 한 문장씩만 압축하면

| STEP                  | 핵심 질문                         |
| --------------------- | ----------------------------- |
| **0. Scope**          | 무엇을 위한 Supervisor를 만드는가?      |
| **1. Capability**     | 실제로 무엇을 보고 사용할 수 있는가?         |
| **2. Baseline**       | 프로젝트는 지금 어떤 상태인가?             |
| **3. Interview**      | Human은 실제로 무엇을 원하는가?          |
| **4. Classification** | 어떤 종류의 프로젝트이며 얼마나 위험한가?       |
| **5. Sufficiency**    | 설계하기에 정보가 충분한가?               |
| **6. Research**       | 외부 근거가 추가로 필요한가?              |
| **7. Architecture**   | 어떤 Supervisor 기능이 필요한가?       |
| **8. Contract**       | 성공·검증·권한을 어떻게 정의할 것인가?        |
| **9. Workflow**       | 어떻게 진행하고 각 작업을 어떻게 통제할 것인가?   |
| **10. Package**       | 이것을 실제 사용 가능한 형태로 어떻게 만들 것인가? |
| **11. Validation**    | 실제로 이 Supervisor가 제대로 작동하는가?  |
| **12. Human Review**  | 최종적으로 Human의 의도와 맞는가?         |

가장 크게 보면 결국:

> **0~3 = 프로젝트를 이해하고 → 4~6 = 어떤 감독이 필요한지 판단하고 → 7~9 = Supervisor를 설계하고 → 10 = 실제 Package로 만들고 → 11~12 = 검증·승인한다**

라고 기억하면 전체 구조가 가장 쉽게 잡힙니다.
