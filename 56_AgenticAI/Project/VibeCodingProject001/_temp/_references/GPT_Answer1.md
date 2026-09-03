전체 그림부터 보면, 이 Workflow의 목적은 **프로젝트를 직접 개발하는 것이 아니라, 해당 프로젝트에 맞는 “Supervisor 운영체계”를 설계·검증해서 활성화하는 것**입니다. 즉 앞부분에서는 프로젝트와 환경을 이해하고, 중간에서는 목표·위험·평가기준·권한·운영방식을 설계하며, 뒷부분에서는 실제 Supervisor Package를 만들고 검증한 뒤 Human 승인 후 가동합니다. 

```text
이해
↓
판단
↓
설계
↓
생성
↓
검증
↓
승인
↓
실제 Supervisor 가동
```

---

## GENERATOR INITIALIZATION

**목적·역할**
Generator 자체를 시작하고, 이번 실행이 어떤 프로젝트를 대상으로 어떤 Supervisor를 만들기 위한 것인지 초기 상태를 잡습니다. 아직 프로젝트를 분석하는 단계라기보다 **Generator 실행 컨텍스트를 여는 단계**입니다.

**왜 필요한가**
Generator 실행과 실제 Supervisor Runtime을 혼동하지 않고, 이후 모든 Step의 기준점이 되는 실행 상태를 만들기 위해 필요합니다.

**예시**

> “현재 `/my-rag-project`를 대상으로 Codex에서 사용할 Supervisor를 생성한다.”

---

## STEP 0 — Generation Contract

**목적·역할**
이번 Generator 실행의 **목표·범위·제약·권한**을 먼저 정의합니다. 무엇을 만들고, 어디까지 조사·파일작성·Research할 수 있는지를 정합니다.

**왜 필요한가**
Generator가 프로젝트 분석 도중 임의로 범위를 확대하거나 민감정보를 외부 Research에 사용하는 등의 문제를 막는 첫 번째 안전 경계입니다. 

**예시**

> “Supervisor Package만 생성하며 Source Code는 수정하지 않는다. Web Research는 가능하지만 사내 코드 원문은 외부 전송 금지.”

---

## STEP 1 — Capability & Environment Discovery

**목적·역할**
현재 Agent가 Repo, Git, Test, Browser, MCP, SubAgent 등을 **실제로 사용할 수 있는지** 확인하고 Evidence 확보 방법을 결정합니다.

**왜 필요한가**
존재하지 않는 기능을 전제로 Supervisor를 설계하면 실행 단계에서 작동하지 않기 때문입니다. 

**예시**

> Codex가 Git/Test/SubAgent를 직접 사용할 수 있음 → 검증을 자동화.
> Test 접근 불가 → Coding Agent에게 Test Result Bundle을 요청하도록 설계.

---

## STEP 2 — Project Baseline / Repo Discovery

**목적·역할**
프로젝트의 현재 상태를 추측하지 않고 **Repo/Git/설정/테스트/문서 기준으로 사실을 확인**하여 Baseline을 만듭니다.

**왜 필요한가**
이후 변경이 원래 있던 것인지 Supervisor가 만든 것인지 구분하고, 실제 State와 Agent의 주장 사이의 불일치를 판단하기 위한 기준점입니다. 

**예시**

> `main` branch, HEAD `abc123`, 기존 미커밋 파일 3개, 테스트 2개 실패 상태를 Baseline으로 기록.

---

## STEP 3 — Adaptive Human Interview

**목적·역할**
Repo만으로 알 수 없는 **Human의 목표·우선순위·성공조건·금지사항·승인범위**를 필요한 만큼만 질문하여 확보합니다.

**왜 필요한가**
기술적으로 올바른 결과와 Human이 원하는 결과는 다를 수 있기 때문입니다. Human의 Goal/Intent/Acceptance는 최상위 판단 기준입니다. 

**예시**

> “속도보다 안정성이 중요합니까?”
> → Human: “기존 기능 Regression이 절대 없어야 합니다.”

---

## STEP 4 — Classification / Risk Modeling

**목적·역할**
프로젝트를 Task Type, Risk, Complexity, Uncertainty, Reversibility 등으로 분류하여 **얼마나 강한 Supervisor가 필요한지 결정**합니다.

**왜 필요한가**
모든 프로젝트에 동일한 Workflow와 검증 수준을 적용하면 단순 작업은 과도하게 무거워지고, 위험한 작업은 통제가 부족해집니다. 

**예시**

> CSS 수정 → Low Risk / High Reversibility
> DB Migration → High Risk / Low Reversibility

---

## STEP 5 — Information Sufficiency Gate

**목적·역할**
지금까지 확보한 정보만으로 Supervisor를 설계할 수 있는지 **중간 Gate에서 판단**합니다.

**왜 필요한가**
핵심 정보가 비어 있는데도 LLM이 임의로 가정하여 Supervisor를 만드는 것을 방지합니다. 부족하면 Interview·Discovery·Research로 돌아갑니다. 

**예시**

> “로그인 개선” 요청은 있으나 기존 인증 방식과 보안 제약을 모름
> → `HOLD` → 추가 Interview.

---

## STEP 6 — Conditional Research & Evidence Synthesis

**목적·역할**
필요한 경우에만 최신 공식 문서·Best Practice·향후 Wiki 등을 조사하고, 이를 **현재 프로젝트에 적합한 Evidence로 변환**합니다.

**왜 필요한가**
LLM 내부 지식만으로 최신 Framework나 보안·Architecture 의사결정을 내리는 위험을 줄이고, 단순 유행이 아니라 프로젝트 적합성을 판단하기 위해 필요합니다. 

**예시**

> 새 Agent Framework 도입 검토
> → 공식 Docs 조사 → 기존 Stack과 호환성 평가 → A는 채택, B는 기각.

---

## STEP 7 — Goal / Evaluation / Authority Contract

**목적·역할**
Supervisor가 앞으로 무엇을 목표로 하고, **무엇을 PASS로 볼지**, 그리고 무엇을 자동 판단하고 언제 Human에게 승인받을지를 명문화합니다.

**왜 필요한가**
Goal, 평가 기준, 권한이 없으면 Supervisor가 “잘 되고 있는지”도 판단할 수 없고 지나치게 많은 권한을 행사할 수도 있습니다. 

**예시**

> Goal: 기존 기능 유지하며 로그인 속도 개선
> Hard Gate: Critical test 100% PASS
> DB 변경: Human Approval 필요.

---

## STEP 8 — Supervisor Architecture Composition

**목적·역할**
앞에서 얻은 프로젝트 특성에 따라 **어떤 Supervisor 기능들을 조합할지 설계**합니다.

**왜 필요한가**
모든 프로젝트에 Research, MCTS, 강한 Handoff, Multi-Agent 등을 전부 넣는 대신 필요한 capability만 선택하기 위해 필요합니다. 

**예시**

> 단순 UI 작업
> → Discovery + Visual Verification + Lightweight Handoff.

> Legacy Bug
> → Repo Discovery + Reproduction + Regression Verification + Recovery.

---

## STEP 9 — Workflow / Gate / Execution-Control Composition

**목적·역할**
실제 프로젝트에서 **어떤 순서로 문제를 풀고**, 각 작업을 어떤 Gate·Evidence·Recovery 규칙으로 통제할지를 조립합니다.

**왜 필요한가**
Supervisor의 Architecture가 “무슨 능력이 있는가”라면, 이 단계는 그 능력을 **실제로 어떤 절차로 사용할지** 결정하는 단계입니다. 

**예시**

> Bug Fix
> `Reproduce → Root Cause → Minimal Fix → Regression Test`

각 단계마다:

> Baseline → Execute → Evidence → Verify → PASS/REVISE/HOLD.

---

## STEP 10 — Supervisor Package Generation

**목적·역할**
지금까지 만든 설계를 실제 Agent가 읽고 사용할 수 있는 **파일·정책·Workflow·State Package로 변환**합니다.

**왜 필요한가**
머릿속 설계나 긴 Prompt가 아니라, Session이 바뀌어도 복구·재사용 가능한 지속적인 Supervisor System으로 만들기 위해 필요합니다. 

**예시**

```text
.vibe-supervisor/
├── PROJECT_PROFILE.md
├── GOAL_CONTRACT.md
├── EVALUATION_PLAN.md
├── CURRENT_STATE.md
├── SUPERVISOR_POLICY.md
├── workflows/
├── research/
└── sessions/
```

---

## STEP 11 — Adversarial Dry Run / Integration Validation

**목적·역할**
생성된 Supervisor를 바로 사용하지 않고, 여러 정상·실패 상황을 가상으로 통과시켜 **실제로 작동 가능한지 공격적으로 검증**합니다.

**왜 필요한가**
문서상 그럴듯하지만 FAIL Recovery, Handoff, Human Approval, Evidence 확보가 실제 상황에서는 작동하지 않는 설계 결함을 미리 찾기 위해 필요합니다. 

**예시**

> Test 실패 → Supervisor가 무한 재시도하지 않고 `REVISE → Root Cause → 재검증`으로 복구하는가?

> SubAgent 사용 불가 → Human-mediated fallback으로 전환되는가?

---

## STEP 12 — Human Acceptance

**목적·역할**
최종적으로 Human이 Goal, Workflow, Risk, Authority, 승인 지점이 실제 의도와 맞는지 확인하고 **Supervisor 활성화 여부를 결정**합니다.

**왜 필요한가**
기술적 검증만으로는 Human의 선호·사업적 판단·허용 가능한 위험 수준까지 확정할 수 없기 때문입니다. 

**예시**

> Human: “Architecture 변경 승인까지 Supervisor가 자동 처리하는 건 싫다.”
> → Authority Contract 수정 후 재검증.

---

# ACTIVATE SUPERVISOR

**목적·역할**
검증과 Human 승인이 끝난 Supervisor Package를 실제 프로젝트의 **Runtime Supervisor로 전환**합니다.

**왜 필요한가**
이 시점부터 Generator의 역할은 끝나고, 생성된 Supervisor가 실제 Coding Agent를 관찰·지시·평가·재계획하는 역할을 맡기 때문입니다. 

**예시**

```text
Generator 종료

        ↓

Supervisor Runtime 시작

Observe Project
→ Determine Next Best Action
→ Instruct Coding Agent
→ Verify Evidence
→ Update State
→ Replan / Handoff
→ Repeat
```

---

### 전체를 한 문장씩 압축하면

| Step                 | 핵심 질문                             |
| -------------------- | --------------------------------- |
| Initialization       | 이번 Generator 실행을 시작할 준비가 됐는가?     |
| **0 Contract**       | 무엇을 어디까지 만들 것인가?                  |
| **1 Capability**     | 실제로 무엇을 할 수 있는가?                  |
| **2 Baseline**       | 지금 프로젝트의 실제 상태는 무엇인가?             |
| **3 Interview**      | Human은 실제로 무엇을 원하는가?              |
| **4 Classification** | 이 프로젝트는 어떤 유형이고 얼마나 위험한가?         |
| **5 Sufficiency**    | Supervisor를 설계할 만큼 정보가 충분한가?      |
| **6 Research**       | 외부 Evidence가 필요한가, 무엇을 채택할 것인가?   |
| **7 Contract**       | 성공·실패·권한을 어떻게 판단할 것인가?            |
| **8 Architecture**   | 어떤 Supervisor 기능이 필요한가?           |
| **9 Workflow**       | 그 기능들을 어떤 절차와 Gate로 운영할 것인가?      |
| **10 Generation**    | 이를 실제 사용 가능한 Package로 어떻게 만들 것인가? |
| **11 Validation**    | 이 Supervisor가 실제 실패 상황에서도 작동하는가?  |
| **12 Acceptance**    | Human이 이 Supervisor를 승인하는가?       |
| **Activate**         | 이제 실제 프로젝트 감독을 시작한다.              |

결국 이 흐름은 **`Understand → Model → Contract → Compose → Generate → Stress-test → Approve → Run`**이라고 볼 수 있습니다. 그리고 Generator는 “프로젝트별 Supervisor를 설계하는 Compiler”, 생성된 Supervisor는 “실제 프로젝트를 지속적으로 통제하는 Runtime”이라는 구분이 핵심입니다. 
