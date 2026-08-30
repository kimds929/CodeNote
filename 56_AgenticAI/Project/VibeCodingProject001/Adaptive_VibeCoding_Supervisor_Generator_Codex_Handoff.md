# Handoff — Adaptive VibeCoding Supervisor Generator Skill

> **Target:** Codex  
> **Purpose:** 이 문서는 ChatGPT에서 논의·확정된 내용을 Codex로 인수인계하여, `Adaptive VibeCoding Supervisor Generator Skill`을 실제로 설계/구현하기 위한 고밀도 handoff 문서다.  
> **Priority:** 이 문서의 **확정된 결정사항(Decisions)** 을 우선 보존하되, 구현 세부는 Codex가 프로젝트/도구 환경을 확인한 뒤 더 나은 구조가 있으면 근거와 trade-off를 제시하고 개선할 수 있다.  
> **Design principle:** 지금 모든 기능을 한 Skill에 구현하지 않는다. 대신 미래의 Planner / Evaluator / Learning / Wiki / Multi-Agent 계층이 **갈아엎지 않고 연결될 수 있는 Interface / Artifact / Schema를 v1부터 설계**한다.

---

# 0. Executive Summary

처음 아이디어는 별도의 창에서 Vibe Coding 대화를 붙여 넣으면 다음 프롬프트를 코칭하는 `VIBECODING_HELPER_AGENT.md`였다.

현재 목표는 그보다 상위 시스템인:

> **Project-Adaptive Vibe Coding Supervisor Generator**

를 만드는 것이다.

이 Skill은 프로젝트 정보를 인터뷰하고, 프로젝트 특성·위험·복잡도·불확실성·사용 Coding Tool·사용 가능한 MCP/도구·기존 LLM Wiki·최신 Web Evidence를 분석한 뒤, **해당 프로젝트에 최적화된 Supervisor Harness**를 생성해야 한다.

생성된 Supervisor는 단순 Prompt Coach가 아니라:

- 프로젝트 전체 Goal/Roadmap을 관리하고,
- 현재 Coding Agent의 답변/행동을 평가하며,
- 잘못된 방향이면 개입하고,
- 현재 State에서 Next Best Action을 판단하고,
- 적절한 다음 Prompt를 만들어 주며,
- 필요하면 Web Research를 수행하고,
- Workflow/Plan을 adaptive하게 수정하고,
- Context Contamination 위험을 감지하여 Session Handoff 시점을 판단하고,
- Handoff 문서와 다음 Session 첫 Prompt를 생성하고,
- 향후에는 다른 SubAgent를 직접 orchestration하는 Supervisor로 확장 가능해야 한다.

전체 시스템의 철학:

```text
Goal = relatively stable
Plan = adaptive
Action = context-dependent
Knowledge = evidence-backed + evolving
Supervisor = decision supervisor, not coding executor
```

---

# 1. Core Problem Definition

Vibe Coding의 핵심 문제는 단순히 “좋은 Prompt를 작성하는 것”이 아니다.

실제 문제는:

1. 프로젝트/Task 종류에 따라 최적 절차가 달라진다.
2. Risk / Complexity / Uncertainty / Reversibility / Project Stage에 따라 Workflow가 달라진다.
3. Agent의 Capability와 Tool(Codex/Claude/Gemini/Kilo 등)에 따라 가능한 운영 방식이 달라진다.
4. 최신 Best Practice가 계속 변한다.
5. 긴 Session은 Context Contamination / Instruction Drift / State Confusion을 일으킬 수 있다.
6. Coding Agent가 말한 내용과 실제 Repo/Test 상태가 다를 수 있다.
7. 프로젝트 종료 결과만 보면 어느 시점의 Supervisor 판단이 좋았거나 잘못됐는지 알기 어렵다.
8. 한 프로젝트의 성공/실패 경험을 다음 프로젝트와 Generator 개선에 재활용하고 싶다.

따라서 목표는 “고정된 Vibe Coding 절차를 만들어 넣는 것”이 아니라:

> **상황에 따라 적절한 Workflow/Method를 선택·조립·평가·개선하는 Supervisor Harness를 프로젝트별로 생성하는 것**

이다.

---

# 2. Confirmed Decisions — 반드시 반영

## 2.1 Tool Independence

### 확정: **C. Portable Core + Tool-Specific Adapter**

완전 Codex 전용도, 완전 Tool-Agnostic도 아니다.

권장 구조:

```text
Portable Supervisor Core
        +
Tool-Specific Adapter
```

예상 구조:

```text
.vibe-supervisor/
├── core / project artifacts
├── workflows/
├── planning/
├── research/
├── telemetry/
├── sessions/
└── adapters/
    ├── codex/
    ├── claude/
    ├── gemini/
    └── kilo/
```

Tool Adapter의 목적:

- Codex → `AGENTS.md`, Codex Skills/도구 특성 반영
- Claude Code → `CLAUDE.md`, permission/session 특성 반영
- Gemini CLI → `GEMINI.md`, checkpoint/recovery 특성 반영
- Kilo Code → `AGENTS.md`, Rules/Workflow/Custom Mode/MCP 특성 반영

**Core methodology는 Tool에 종속시키지 않는다.**

향후 새로운 Coding Tool이 생기면 Adapter만 추가 가능해야 한다.

---

## 2.2 Supervisor Input

### 확정: **C. Hybrid Input / Evidence Adapter**

현재 초기 사용 방식은 사용자가 Coding Agent와의 대화 내용을 별도 Supervisor Session에 붙여 넣는 manual 방식일 수 있다.

그러나 미래에는 Supervisor Agent가 다른 SubAgent를 자동 관리하도록 확장할 계획이 있으므로, 처음부터 다음 입력원을 고려한다.

```text
Manual Conversation Paste
Coding Agent Output / Session Log
Repository State
Git diff / git log
Test / Build / Lint results
Project Artifact
Tool execution logs
Human feedback
Future SubAgent messages
```

Input Adapter 개념을 고려:

```text
ManualInputAdapter
RepositoryAdapter
GitAdapter
SessionLogAdapter
ToolLogAdapter
SubAgentAdapter
```

### Source-of-Truth / Evidence Priority

서로 충돌하면 “Agent가 말했다”보다 실제 실행 증거를 우선한다.

권장 우선순위:

```text
Executable Evidence
(Test / Build / Runtime)
>
Repository / Git State
>
Versioned Project Artifacts
>
Tool Execution Logs
>
Coding Agent Statements
>
Human Summary
```

단, Human의 **Goal / Intent / Acceptance 판단**은 별도 최상위 권한을 가진다.

---

## 2.3 Supervisor Authority

### 확정: **C. Tiered Authority**

Supervisor를 Advisory-only로 제한하지 않지만, Source Code까지 마음대로 수정하는 Full Authority도 주지 않는다.

기본 원칙:

> **Supervisor는 Coding Agent가 아니다. Supervisor는 상태/계획/평가를 관리하는 감독자다.**

권장 권한:

| Artifact / Action | 권한 |
|---|---|
| `CURRENT_STATE` | 자동 Update |
| Session Handoff | 자동 생성/Update |
| Research Ledger | 자동 Update |
| Decision / Reward / Evaluation logs | 자동 Update |
| ROADMAP 진행률 | 자동 Update |
| ROADMAP의 구조적 변경 | 조건부 / 중요 시 Human Approval |
| Workflow의 Low-risk 개선 | 자동 가능 |
| Workflow의 Medium-risk 변경 | 다음 Task/Phase부터 |
| High-risk 방법론/Architecture 정책 변경 | Human Approval |
| Architecture Decision | Human Approval 권장 |
| Project Source Code | 기본 Read-only |
| Coding Agent에게 줄 Prompt/Instruction | Supervisor 핵심 권한 |

Supervisor write area는 기본적으로 `.vibe-supervisor/**` 같은 관리 영역으로 한정하는 방향을 우선 검토한다.

---

## 2.4 Web Research Trigger

### 확정: **D. Event-Driven + Freshness TTL Hybrid**

Web Search는 보조 기능이 아니라 이 Generator의 핵심 차별점이다.

목표:

> LLM의 기억/자체 판단에만 의존하지 않고, 프로젝트 종류에 맞는 최신 Best Practice/공식 문서/방법론을 적극 조사해 Supervisor를 생성한다.

### Research Trigger 후보

- Project Initialization
- Phase Change
- 신규 Framework / Library / Tool 도입
- Architecture Decision
- High-risk / irreversible decision
- Repeated Failure
- 기존 Plan과 실제 결과의 큰 불일치
- Supervisor confidence 저하
- Requirement ambiguity
- Scope Drift
- 기존 Workflow 성능 저하
- 기존 Research evidence가 stale
- HUMAN이 최신 방법론 검증을 요청

### Freshness TTL 개념

정보 종류별로 유효기간을 다르게 두는 방향:

```text
빠르게 변하는 Agent/Tool Best Practice → 짧은 TTL
Framework / Library Practice         → 중간 TTL
일반 SE Methodology                  → 긴 TTL
```

정확한 TTL 값은 hard-code하지 말고 config/adaptive policy로 설계한다.

### Source Priority

```text
1. Official Docs / Official Engineering
2. Official Repository / Maintainer Docs
3. Research Paper / Maintainer Research
4. Reputable Engineering Blog
5. Community / Blog / Reddit / anecdotal cases
```

Research 결과는 바로 Workflow에 주입하지 말고:

```text
Search
→ Source Classification
→ Evidence Evaluation
→ Project-Fit Evaluation
→ Adoption Decision
→ Research Ledger
```

과정을 거친다.

---

## 2.5 Existing LLM Wiki Integration

향후 `LLM Wiki`를 별도 Knowledge System으로 확장할 계획이다.

Generator/Supervisor는 Web뿐 아니라 Wiki를 조회할 수 있어야 한다.

역할 구분:

```text
Web = 최신 External Knowledge
Wiki = 누적된 Internal / Cross-Project Knowledge
```

Wiki에 현재 프로젝트 방향과 관련된 기존 지식/Playbook/Decision이 있으면 **자동 적용하지 말고**, 프로젝트 영향이 있는 경우 HUMAN에게:

> “기존 Wiki에 이 유형 프로젝트에서 X를 사용한 기록/권장사항이 있습니다. 이번 프로젝트에도 적용할까요?”

처럼 제안해야 한다.

Wiki는 Generator 본체에 구현하지 않는다.  
**Wiki Integration Interface / Retrieval Contract를 지금 설계**한다.

---

## 2.6 Handoff Trigger

### 확정: **D. Multi-Signal Semantic Handoff**

단순 Turn 수나 Token 수만으로 Session을 끊지 않는다.

Supervisor가 프로젝트 흐름을 보고 **적절한 semantic boundary에서 알아서 Handoff를 권고/수행**해야 한다.

### Hard Signals 후보

- Context pressure가 위험 수준
- Agent가 이전 결정/State를 반복 혼동
- Contradictory state 발견
- Phase / Milestone 완료
- 현재 Session의 목표 완료
- 다음 Task가 독립적이며 새 Session이 유리
- Critical decision 이후 clean context가 유리
- Tool이 제공하는 context 사용량 임계치 도달

### Soft Signals 후보

- Topic Drift 증가
- Debugging history 과도 누적
- 과거 내용을 반복 설명
- Instruction contamination 의심
- 새로운 목표가 기존 Session 목표와 달라짐
- Agent response consistency 저하
- 많은 실패/rollback/history가 누적되어 current state보다 historical noise가 커짐

Supervisor는 매 Turn:

```text
CONTINUE
or
HANDOFF RECOMMENDED
or
HANDOFF REQUIRED
```

를 판단할 수 있어야 한다.

### Handoff 시 생성할 것

```text
HANDOFF_<session-id>.md
```

포함 정보:

- Project Goal
- Current Phase
- Active Task
- Completed Work
- Current Repo State
- Important Decisions
- Constraints
- Tests / Build status
- Known Issues
- Unresolved Questions
- Failed Attempts (필요한 것만)
- Current Hypotheses
- Risks
- Next Best Action
- Recommended Workflow Stage
- Required files to read
- Explicit “do not inherit” stale context
- Next Session Bootstrap Prompt

핵심:

> **Handoff 문서 전체를 다음 Prompt에 다시 복붙해서 Context Contamination을 재생산하지 않는다.**

Progressive Disclosure를 사용:

```text
AGENTS/Adapter
→ CURRENT_STATE
→ 필요한 Roadmap
→ 필요한 Decisions
→ 필요한 과거 Handoff
```

Next Session 첫 Prompt에는 **필요한 최소 정보 + 읽어야 할 파일 위치**만 제공하는 방향을 우선 검토한다.

---

## 2.7 Quality Evaluation

### 확정: **D. Hard Gate + Multi-Dimensional Scorecard**

단일 총점만 사용하지 않는다.

### Hard Gates 예시

- Build success
- Critical tests pass
- Regression gate
- Critical requirements
- Security/Integrity gate (project-dependent)

### Multi-dimensional Evaluation 예시

```text
Correctness
Requirement Satisfaction
Verification Quality
Architecture Fit
Maintainability
Human Intent Alignment
Scope Control
Rework
Human Correction Count
Efficiency / unnecessary turns
Workflow Fit
Risk Management
Documentation / State consistency
```

원본 평가는 Vector 형태로 저장한다.

예:

```yaml
evaluation:
  requirement_satisfaction: 0.94
  tests: 1.00
  architecture_fit: 0.78
  human_alignment: 0.91
  scope_control: 0.83
  rework_efficiency: 0.65
```

**Planner/MCTS가 scalar가 필요할 때만 context-specific weights로 scalarize**한다.

즉:

```text
Evaluation = multidimensional truth
Planner Value = context-specific scalar projection
```

단일 숫자만 저장하지 않는다.

---

## 2.8 Method Update Policy

### 확정: **D. Risk-Tiered + Versioned + Phase-Safe Update**

새 Best Practice를 발견했다고 현재 Workflow를 즉시 갈아끼우지 않는다.

권장:

### Low Risk
예:
- Prompt wording
- Handoff template
- Research query
- telemetry format minor improvement

→ 자동 적용 가능

### Medium Risk
예:
- Workflow step order
- validation step 추가
- bugfix process 개선

→ 다음 Task / 다음 Phase부터 적용

### High Risk
예:
- TDD 도입
- Architecture methodology 변경
- Branching strategy 변경
- Security policy 변경
- DB migration strategy 변경
- Tool/Framework 교체

→ Human Approval

모든 Method 변경은 versioning:

```text
feature-workflow v1.2 → v1.3
```

그리고 기록:

```text
why_changed
evidence
expected_benefit
affected_scope
rollback_condition
```

---

# 3. Scope Boundary — 가장 중요한 결정

사용자는 장기적으로 다음을 모두 원한다.

- Supervisor Generator
- Supervisor Runtime
- MCTS Planning
- Session/Phase/Project Evaluation
- Reward + Bellman/TD Credit Assignment
- Eligibility Trace
- LLM Wiki
- Cross-project learning
- Generator improvement
- Multi-Agent / SubAgent orchestration

그러나 **모든 기능을 Generator Skill 하나에 구현하면 안 된다.**

권장 구조:

```text
                    HUMAN
                      │
                      ▼
┌────────────────────────────────────┐
│ ① SUPERVISOR GENERATOR             │
│ Adaptive VibeCoding Supervisor     │
│ Generator Skill                    │
└─────────────────┬──────────────────┘
                  │ generates
                  ▼
┌────────────────────────────────────┐
│ ② SUPERVISOR RUNTIME               │
│ 실제 프로젝트 감독                 │
└──────────────┬──────────────┬──────┘
               │              │
               ▼              ▼
      ③ PLANNER ENGINE    Coding/SubAgents
      Direct/Simple/
      Research/MCTS
               │
               ▼
┌────────────────────────────────────┐
│ ④ EVALUATION ENGINE                │
│ Session / Phase / Project Eval     │
└─────────────────┬──────────────────┘
                  ▼
┌────────────────────────────────────┐
│ ⑤ LEARNING / CREDIT ASSIGNMENT     │
│ Reward / TD(λ) / Eligibility Trace │
└─────────────────┬──────────────────┘
                  ▼
┌────────────────────────────────────┐
│ ⑥ LLM WIKI / KNOWLEDGE SYSTEM      │
│ Cross-project Knowledge            │
└─────────────────┬──────────────────┘
                  ▼
          Generator Evolution
```

### v1의 핵심 원칙

> **지금은 ① Generator를 제대로 만들되, ②~⑥이 나중에 연결될 Socket / Interface / Schema를 v1부터 만든다.**

---

# 4. Adaptive Supervisor Generator v1 — In Scope

Generator는 최소 다음을 담당해야 한다.

## 4.1 Project Interview / Discovery

프로젝트에 필요한 정보를 **고정 질문 + Adaptive 질문**으로 수집.

최소 후보:

```text
Project Goal
Success Definition
Project Type
Domain
Greenfield / Existing
Current Project Stage
Tech Stack
Repository Structure
Critical Constraints
Risk
Complexity
Uncertainty
Reversibility
Human Priority
Human preferred working style
Coding Tool
Available MCP / Tools
Web availability
Existing Project Docs
Existing LLM Wiki
Security / privacy constraints
Deployment target
Testing availability
Evaluation feasibility
Timeline / Cost constraints
```

질문은 무조건 모두 던지는 방식보다, 기존 답변과 Repo/문서를 보고 필요한 것만 묻는 Adaptive Interview가 목표.

---

## 4.2 Project Classification

예시 taxonomy:

```text
New Feature
Bug Fix
Refactor
Migration
Research
Prototype
Performance
Security
Data / ML
RAG / Agentic AI
Frontend / UI
Backend
Infrastructure
Deployment
Legacy Understanding
etc.
```

그리고 최소:

```text
Task Type
Risk
Complexity
Uncertainty
Reversibility
Project Stage
```

를 Workflow 선택/조립 기준으로 사용.

---

## 4.3 Research Synthesis

Web + Official docs + 향후 Wiki를 활용하여:

```text
Research Questions
→ Search
→ Evidence
→ Project Fit
→ Selected Practice
→ Rejected Alternatives
→ Confidence
→ Research Ledger
```

를 남긴다.

Research 결과는 “유행”이 아니라 프로젝트에 맞는지 평가해야 한다.

---

## 4.4 Project Evaluation Contract — 필수

프로젝트 실행 전에 **Project Goal의 정량/정성 평가 계획서**를 만든다.

권장 Artifact:

```text
GOAL_CONTRACT.md
EVALUATION_PLAN.md
```

내용:

```text
Goal
Success criteria
Critical requirements
Hard gates
Quality dimensions
Human acceptance criteria
Measurement method
Evidence source
When to evaluate
Who can evaluate
Terminal conditions
```

가능한 경우 정량화:

```yaml
success_criteria:
  functional:
    critical_features: 100%
  quality:
    critical_test_pass: 100%
    regression: 0
  human_alignment:
    acceptance_target: ">= 4/5"
```

주의:

- 모든 프로젝트에 억지로 숫자를 넣지 않는다.
- 평가 가능한 것만 계측한다.
- 정량 + 정성 평가를 함께 허용한다.
- 평가 기준은 프로젝트 중간에 함부로 바꾸지 않으며 변경 이력을 남긴다.

---

## 4.5 Supervisor Harness Generation

Generator의 핵심 Output은 **단일 `AGENTS.md`가 아니라 Project-specific Supervisor Package**다.

권장 최소 구조:

```text
.vibe-supervisor/
│
├── MANIFEST.yaml
├── PROJECT_PROFILE.md
├── GOAL_CONTRACT.md
├── EVALUATION_PLAN.md
├── ROADMAP.md
├── CURRENT_STATE.md
├── SUPERVISOR_POLICY.md
│
├── policies/
│   ├── authority.md
│   ├── research.md
│   ├── handoff.md
│   ├── knowledge.md
│   └── evaluation.md
│
├── planning/
│   ├── planner-policy.md
│   ├── action-space.yaml
│   ├── reward-model.yaml
│   └── search-config.yaml
│
├── workflows/
│   └── project-specific workflows
│
├── playbooks/
│   └── project-specific playbooks
│
├── research/
│   └── RESEARCH_LEDGER.md
│
├── decisions/
│   └── ADR / decision records
│
├── telemetry/
│   ├── DECISION_LOG.jsonl
│   ├── REWARD_EVENTS.jsonl
│   └── EVALUATION_LOG.jsonl
│
├── sessions/
│   ├── ACTIVE_SESSION.md
│   └── handoffs/
│
└── adapters/
    └── selected coding-tool adapter
```

그리고 Tool-specific entry point:

```text
Codex   → AGENTS.md
Claude  → CLAUDE.md
Gemini  → GEMINI.md
Kilo    → appropriate AGENTS/rules/workflows/mode config
```

**Entry file은 brain이 아니라 map/index 역할**을 해야 한다.

---

# 5. Workflow Philosophy

고정 Workflow 하나를 모든 Task에 강제하지 않는다.

일반적인 공통 뼈대는 있을 수 있지만:

```text
Explore
Plan
Execute
Verify
Review
Update
```

실제 절차는 다음 함수처럼 생각한다.

```text
Workflow =
f(
  Task Type,
  Complexity,
  Risk,
  Uncertainty,
  Reversibility,
  Project Stage,
  Existing Codebase,
  Agent Capability,
  Tool Capability
)
```

권장 방향은 “Workflow 파일의 단순 선택”을 넘어서 **Workflow Module / Grammar**로 확장 가능하게 설계.

예:

```text
workflow-modules/
├── explore
├── reproduce
├── specify
├── architect
├── research
├── plan
├── prototype
├── implement
├── unit-test
├── integration-test
├── review
├── benchmark
├── security-check
├── handoff
└── deploy
```

각 Module에는 최소:

```text
Use when
Avoid when
Inputs
Outputs
Exit criteria
Evidence requirements
Risk
```

metadata를 둘 수 있다.

예:

```text
Bug Fix + High uncertainty
→ Reproduce → Explore → Root Cause → Plan → Fix → Regression Test

UI Prototype + Low risk
→ Requirement → Prototype → Visual Verify → Iterate
```

---

# 6. Supervisor Runtime — Generator가 생성해야 할 행동 계약

생성된 Supervisor는 매 Turn/Observation마다 대략 다음 Loop를 수행한다.

```text
1. Read/Recover Project Goal
2. Reconstruct Current State
3. Compare Actual State vs Plan
4. Inspect Coding Agent output + actual evidence
5. Detect:
   - uncertainty
   - risk
   - failure
   - scope drift
   - requirement drift
   - verification gaps
   - state contradiction
   - session contamination risk
6. Decide planning mode
7. Determine Next Best Action
8. Produce next instruction/prompt
9. Observe actual execution result
10. Validate
11. Update State
12. Update Roadmap if allowed/necessary
13. Log decision/reward/evaluation events
14. Decide CONTINUE vs HANDOFF
15. Repeat
```

Supervisor의 output은 최소:

```text
Current State
Diagnosis
Detected Risks / Drift
Current Workflow Stage
Recommended Next Best Action
Why
Required Evidence
Prompt to send to Coding Agent
Handoff status
```

형태를 고려.

---

# 7. MCTS / Adaptive Planning — 핵심 설계

첨부된 기존 MCTS handoff의 결론을 그대로 계승한다.

## 7.1 Pure MCTS가 아니다

권장:

> **Hierarchical Planner + Rolling-Horizon MCTS + Actual Environment Feedback**

```text
Plan
→ Search
→ Best Next Action
→ Execute ONLY first action
→ Observe Actual Result
→ Update State
→ Replan
```

전체 미래를 확정하지 않는다.

ROADMAP은:

> **Living Hypothesis / best-known plan**

으로 취급한다.

---

## 7.2 Hierarchical Planning

```text
Level 1  PROJECT GOAL
Level 2  PHASE PLAN
Level 3  DECISION / SEARCH
         Direct / Simple Plan / Research / MCTS
Level 4  NEXT BEST ACTION
Level 5  CODING AGENT EXECUTION
Level 6  VALIDATION / ENVIRONMENT FEEDBACK
Level 7  STATE UPDATE + REPLAN
```

MCTS는 Supervisor 전체가 아니라 **Decision Engine의 한 Planning Mode**다.

---

## 7.3 Adaptive Search Gate

모든 Turn에 MCTS를 실행하면 비용이 과도하므로 Search 필요 여부를 먼저 결정한다.

예:

```text
High confidence + Low risk
→ Direct

명확한 대안 2~3개
→ Simple Comparison

Medium uncertainty
→ Shallow Search

High uncertainty / High impact
→ MCTS

Repeated Failure
→ Deep Search + Research

Architecture Decision
→ Research / MCTS

New Technology
→ Research + MCTS
```

MCTS Trigger 후보:

- 여러 구현 방법이 경쟁
- Architecture 변경
- New library/framework
- 반복 실패
- Plan vs actual 큰 불일치
- Requirement ambiguity
- Scope drift
- Large refactor
- irreversible/high-cost decision
- Human correction 반복

MCTS 불필요 후보:

- 단순 rename
- 명확한 CSS 수정
- 반복 CRUD
- 원인이 확실한 bug fix
- 이미 검증된 pattern

---

## 7.4 MCTS State

State는 대화 전체가 아니라 **구조화된 project state**여야 한다.

예:

```yaml
state:
  project_goal:
  current_phase:
  active_task:
  completed_work:
  current_repo_state:
  failing_tests:
  unresolved_issues:
  architectural_constraints:
  human_intent:
  roadmap:
  recent_actions:
  observed_results:
  confidence:
  risks:
```

### 반드시 구분

```text
Simulated State
!=
Observed State
```

LLM 예상과 실제 환경을 섞지 않는다.

```yaml
simulated_state:
  predicted_outcome:
  expected_risks:
  expected_test_result:

observed_state:
  actual_test_result:
  actual_build_result:
  actual_repo_changes:
  human_feedback:
```

---

## 7.5 MCTS Action Space

Action은 코드 수정만이 아니다.

예:

```text
Analyze existing code
Ask HUMAN
Web Research
Compare library/framework
Write tests first
Prototype
Design interface
DB schema change
Adapter introduction
Refactor
Rollback
Continue current approach
Switch implementation strategy
Request evaluator
Request subagent
Handoff session
```

미래 Multi-Agent에서는 Action이 특정 SubAgent invocation이 될 수 있다.

---

# 8. Evaluation / Reward / Learning

## 8.1 MCTS Backpropagation과 Bellman/TD Credit Assignment는 분리

두 개는 역할이 다르다.

### MCTS Backpropagation

현재 Search Tree에서 **simulated candidate paths**의 가치를 부모 노드에 반영.

### TD / Bellman-inspired Credit Assignment

실제로 프로젝트가 진행된 뒤:

```text
S1-A1
→ S2-A2
→ S3-A3
→ ...
→ Reward / Final Outcome
```

에서 **과거 어느 Decision이 현재 결과에 기여했는지** 평가.

동일한 “backup” 용어를 쓰더라도 별도 subsystem으로 설계한다.

---

## 8.2 Eligibility Trace 아이디어

Reward가 발생할 때 직전 행동만 평가하지 않고, 이전 여러 State/Action에도 credit을 전달한다.

초기 개념:

```text
Credit
=
Eligibility
×
Causal Relevance
×
Evidence Confidence
```

단순 시간 proximity만 사용하지 않는다.

예:
프로젝트 성공 직전 UI 색상 수정은 시간상 최근이지만 architecture 성공에는 causal relevance가 낮다.

---

## 8.3 Reward Event는 Project 종료에만 두지 않는다

가능한 event:

| Event | Type |
|---|---|
| Test pass/fail | Objective |
| Build pass/fail | Objective |
| Regression | Objective negative |
| Requirement complete | Objective/Semi |
| Human correction | Human feedback |
| Human acceptance | Human reward |
| Architecture approval | Human/Evaluator |
| Repeated failure | Negative |
| Phase complete | Aggregate |
| Session end | Intermediate |
| Project end | Terminal |

Loop:

```text
Action
→ Observation
→ Reward Event
→ Local Credit Update

Session End
→ Session Evaluation
→ Trace Update

Project End
→ Final Multi-dimensional Evaluation
→ Long-range Credit Assignment
```

---

## 8.4 초기에는 “진짜 RL 학습”으로 과장하지 않는다

Software project environment는 non-stationary하다.

변화 요소:

```text
Human intent
Repository
Framework versions
Coding model
Supervisor model
Tool capability
Web information
Constraints
```

따라서 v1/v2에서는:

> **TD(λ)-inspired Credit Assignment / Learning Telemetry**

로 시작하는 것이 안전하다.

충분한 데이터가 쌓인 뒤 `V(S)`, `Q(S,A)`, learned policy를 검토.

---

# 9. Decision / Reward / Evaluation Telemetry — v1부터 준비

향후 학습을 위해 **Decision Trace를 반드시 남긴다.**

예:

```yaml
decision:
  id: D-0182
  session_id: SESS-07
  state_id: ST-0178
  phase: feature-development

  state_summary:
    uncertainty: high
    risk: medium

  candidate_actions:
    - investigate
    - refactor
    - prototype

  selected_action: investigate
  planning_mode: shallow_mcts

  rationale:
    - root-cause uncertainty high

  expected_result:
    dependency_conflict_identified: true

  confidence: 0.71
  evidence_refs: []
```

Observation:

```yaml
observation:
  decision_id: D-0182

  actual_result:
    dependency_conflict_found: true

  evidence:
    test:
    build:
    git_diff:

  reward_events:
    - type: test_pass
      value:
    - type: rework_avoided
      value:

  human_feedback:
    accepted: true
```

v1에서는 저장만 해도 된다.  
나중에 Learning Skill이 이를 소비하도록 schema 안정성을 중요하게 본다.

---

# 10. Project Evaluation — 누가 평가하는가?

마지막 평가는 여러 방식 가능:

```text
HUMAN
Project Evaluator Skill
Evaluator Agent
Hybrid
```

권장 Future Architecture:

```text
Objective evidence
+ Project Evaluator
+ HUMAN acceptance
```

Evaluator가 생성기와 같은 model self-evaluation만 반복하지 않도록 가능한 경우 별도 evaluator / independent criteria / executable evidence를 사용.

---

# 11. LLM Wiki / Cross-Project Knowledge

Wiki는 별도 subsystem.

모든 성공/실패를 바로 Global Rule로 올리지 않는다.

Knowledge maturity:

```text
OBSERVATION
↓
PROJECT LESSON
↓
REPEATED PATTERN
↓
VALIDATED PRACTICE
↓
RECOMMENDED DEFAULT
```

예:

한 프로젝트에서 “TDD가 좋았다”
→ project observation

여러 유사 프로젝트에서 반복 성공
→ cross-project pattern

Evaluator/Human 검증
→ recommended practice

Wiki에 저장할 만한 의미 있는 정보:

- 반복 실패 패턴
- 성공 workflow
- tool-specific caveat
- architecture lessons
- high-value prompts
- handoff patterns
- context failure cases
- Research evidence
- project type별 best practice
- supervisor failure pattern
- evaluation metric validity
- reward shaping lessons

---

# 12. Generator Evolution / Meta-Learning

Generator가 한 프로젝트의 feedback을 받고 **자기 Skill을 즉시 self-modify하면 안 된다.**

권장:

```text
Project Evaluation
↓
Credit Assignment
↓
Root Cause Analysis
↓
GENERATOR_IMPROVEMENT_PROPOSAL
↓
Evidence accumulation across projects
↓
Human / Meta Evaluator
↓
Generator version update
```

예:

```yaml
generator_improvement_proposal:
  candidate_change:
    "DB migration project interview에 rollback strategy 질문 추가"

  evidence:
    - project_18
    - project_23
    - project_31

  confidence: high
  risk: low
```

두 Learning Loop를 구분:

## Inner Loop — 프로젝트 내부

```text
State
→ Plan
→ Action
→ Environment
→ Reward
→ State Update
→ Replan
```

## Outer Loop — 프로젝트 간

```text
Projects
→ Evaluation
→ Credit Assignment
→ Pattern Extraction
→ Wiki
→ Generator Improvement
→ Generator vNext
```

---

# 13. Multi-Agent / Autonomous Future Extension

현재는:

```text
User
↔ Coding Agent Window
↔ Supervisor Window
```

의 manual orchestration일 수 있다.

미래에는:

```text
Supervisor
├── Research Agent
├── Code Explorer Agent
├── Architecture Agent
├── Coding Agent
├── Test Agent
└── Evaluator Agent
```

처럼 자동 orchestration하는 구조까지 고려.

따라서 v1 schema/config에:

```yaml
execution_mode:
  current: manual
  supported_future:
    - assisted
    - autonomous_multi_agent
```

같은 개념을 고려.

Supervisor가 Agent invocation을 Action으로 다룰 수 있게 Action schema를 지나치게 Prompt-only로 고정하지 않는다.

예:

```yaml
action:
  type: prompt_handoff | research | tool_call | subagent_call | human_question | session_handoff
  target:
  payload:
  expected_evidence:
```

---

# 14. Recommended Artifact Responsibilities

## `MANIFEST.yaml`

Machine-readable package overview.

예상 필드:

```yaml
supervisor_package_version:
generator_version:
project_id:
project_type:
coding_tool:
execution_mode:
created_at:
last_updated:
artifacts:
capabilities:
integrations:
```

---

## `PROJECT_PROFILE.md`

상대적으로 stable한 project facts / constraints / environment.

---

## `GOAL_CONTRACT.md`

Goal과 성공 정의.

Goal을 바꿀 경우 반드시 Human authority와 change log 고려.

---

## `EVALUATION_PLAN.md`

어떻게 성공/실패를 판단하는지.

Hard gate + Vector Evaluation.

---

## `ROADMAP.md`

절대 명령서가 아니라:

> **현재 시점의 best-known plan / living hypothesis**

변경 가능하되 version/change reason 필요.

---

## `CURRENT_STATE.md`

Supervisor가 매 turn/session 복구할 핵심 compact state.

대화 history의 대체물이 되어야 한다.

---

## `SUPERVISOR_POLICY.md`

Supervisor Runtime의 핵심 loop / response contract / evidence priority / anti-drift 원칙.

---

## `policies/authority.md`

어떤 artifact/action을 자동 변경하고 언제 Human approval이 필요한지.

---

## `policies/research.md`

Event Trigger + TTL + source priority + evidence gate.

---

## `policies/handoff.md`

Multi-signal handoff 판단, 생성 artifacts, next-session bootstrap contract.

---

## `policies/knowledge.md`

Web/Wiki 사용, knowledge maturity, Human confirmation policy.

---

## `policies/evaluation.md`

Hard gates, dimensions, evaluator sources, score handling.

---

## `planning/planner-policy.md`

Direct / Simple / Research / MCTS 선택 정책.

---

## `planning/action-space.yaml`

Supervisor가 허용하는 Action의 machine-readable registry.

---

## `planning/reward-model.yaml`

Evaluation dimension / planner weights / reward event schema.

주의:
Evaluation truth와 planner scalar value를 분리.

---

## `planning/search-config.yaml`

MCTS budget / depth / branch / rollout strategy / research integration.

v1에서 full MCTS engine을 구현하지 않아도 config schema는 준비.

---

## `research/RESEARCH_LEDGER.md`

각 Research:

```text
Question
Search date
Source
Evidence level
Finding
Project fit
Adopted / Rejected
Reason
Confidence
Freshness
```

---

## `telemetry/DECISION_LOG.jsonl`

Decision tracing.

---

## `telemetry/REWARD_EVENTS.jsonl`

Reward events.

---

## `telemetry/EVALUATION_LOG.jsonl`

Session/Phase/Project evaluation.

---

## `sessions/ACTIVE_SESSION.md`

이번 session의 목표/현재 state/allowed scope.

---

## `sessions/handoffs/`

Context reset을 위한 인수인계.

---

# 15. Critical Architectural Principles

## 15.1 AGENTS.md / CLAUDE.md 등 Entry Instruction을 비대하게 만들지 않는다

Entry file의 역할:

```text
Role
Goal
Critical Rules
Where to read
How to resume
Tool adapter behavior
```

실제 상세 knowledge/workflow/state는 `.vibe-supervisor/`에 둔다.

---

## 15.2 Progressive Disclosure

모든 자료를 context에 한꺼번에 넣지 않는다.

```text
Entry
→ Current State
→ Active Workflow
→ Necessary Roadmap section
→ Relevant Decision
→ Relevant Research
→ Historical Handoff only if needed
```

---

## 15.3 Observed Evidence > Agent Narrative

“테스트 성공”이라고 말한 것보다 실제 test output 우선.

---

## 15.4 Simulated State ≠ Observed State

MCTS/LLM 예상과 실제 실행 결과를 섞지 않는다.

---

## 15.5 Supervisor ≠ Coding Agent

Supervisor는 기본적으로:

```text
Observe
Diagnose
Research
Plan
Evaluate
Direct
Handoff
Update supervisor artifacts
```

Coding Agent는:

```text
Implement
Modify source
Run
Test
```

역할을 담당.

미래 Multi-Agent에서는 경계가 더 중요하다.

---

## 15.6 Goal Stable, Plan Adaptive

Human Goal/Success Contract는 상대적으로 안정적.

Plan/Workflow/Next Action은 evidence에 따라 adaptive.

---

## 15.7 Evaluation Raw Data를 보존

현재 weight로 계산한 scalar score만 저장하지 않는다.

향후 Reward Model을 바꾸어도 재평가 가능하게 raw metrics/evidence 보존.

---

# 16. v1 In-Scope / Out-of-Scope / Future Socket

## v1 In-Scope

- Generator Skill
- Adaptive Project Interview
- Project Classification
- Web Research policy/integration
- Existing Wiki integration contract
- Project Evaluation Contract 생성
- Tool Core + Adapter 구조
- Project-specific Supervisor Harness 생성
- Workflow/Playbook 생성
- Supervisor Runtime Protocol 생성
- State / Decision / Reward / Evaluation schema
- Session Handoff policy + template + bootstrap prompt
- Tiered authority
- Research Ledger
- MCTS planning policy/config/interface
- Planner trigger policy
- telemetry files/schema
- Method versioning/update policy
- Multi-Agent future-compatible Action schema

## v1에서 “직접 고도 구현하지 않아도 되는 것”

- Full MCTS rollout engine
- Actual RL policy learning
- Q/V function training
- Cross-project Bellman optimization
- Global Wiki implementation
- Automatic Generator self-rewrite
- Fully autonomous multi-agent orchestration
- Sophisticated causal credit assignment model

## 그러나 v1에서 반드시 Interface/Schema 준비

- Planner Engine interface
- Evaluator interface
- Reward Event interface
- Learning/Credit Assignment interface
- Wiki retrieve/store candidate interface
- SubAgent action interface
- Generator improvement proposal interface

---

# 17. Open Design Questions — Codex가 구현 전 검토할 것

아래는 아직 “구현 세부 확정”이 필요한 영역이다.

## A. Skill 자체의 설치 위치 / 호출 방식

- Global skill?
- Project-local skill?
- skill name / trigger
- 생성 대상 프로젝트 root를 어떻게 받는지

## B. Exact Generator Workflow

권장 기본:

```text
Intake
→ Inspect existing repo/docs
→ Interview
→ Project classification
→ Research plan
→ Web/Wiki research
→ Evidence synthesis
→ Goal/Evaluation contract
→ Architecture of supervisor
→ Workflow composition
→ Artifact generation
→ Validation
→ Human review
→ Finalize
```

이를 더 세분화해야 함.

## C. Interview Termination Criteria

“충분한 정보가 모였는지”를 어떻게 판단할지.

## D. Research Budget

query count, token/time budget, source diversity, stop rule.

## E. Workflow Module Schema

Markdown only vs YAML metadata + Markdown body.

## F. State Schema

최소 state와 project-specific extension을 어떻게 분리할지.

## G. Decision Log Schema

JSONL/YAML/Markdown 중 machine readability와 human readability balance.

## H. Handoff Need Score

Hard/soft signal을 rule-based로 시작할지 LLM judgement + heuristic hybrid로 할지.

## I. MCTS Search Budget

branch / depth / rollout / tool budget.

## J. Evaluator Contract

HUMAN / same LLM / independent Agent / executable evidence의 결합 방식.

## K. Reward Event Scale

절대 수치보다 normalized vector가 더 적절한지 검토.

## L. Causal Relevance

Eligibility trace에 causal relevance를 나중에 어떤 방식으로 추가할지.

## M. Wiki Knowledge Promotion

Observation → Validated Practice 승격 조건.

## N. Generator Improvement Proposal

한 프로젝트 feedback이 global generator에 과도하게 반영되지 않도록 evidence threshold.

---

# 18. Recommended Implementation Order for Codex

## Phase 1 — Architecture

1. 이 handoff를 읽고 요구사항을 재구성
2. In-scope/out-of-scope 확인
3. Generator architecture 제안
4. Artifact dependency graph 작성
5. Schema/interface 설계
6. Tool Adapter boundary 설계

**아직 Skill.md를 바로 길게 작성하지 말 것.**

---

## Phase 2 — Minimum Generator Contract

먼저 다음 6개 핵심을 정의:

```text
1. Generator Input
2. Interview protocol
3. Research protocol
4. Supervisor Package output
5. Runtime contract
6. Extension interfaces
```

---

## Phase 3 — Artifact Schemas

우선:

```text
MANIFEST
PROJECT_PROFILE
GOAL_CONTRACT
EVALUATION_PLAN
CURRENT_STATE
ROADMAP
DECISION_LOG
REWARD_EVENT
HANDOFF
```

를 정의.

---

## Phase 4 — Skill Workflow

그 다음 실제 `SKILL.md` 작성.

Skill은 “정답 template을 복사하는 generator”가 아니라:

```text
Inspect
→ Interview
→ Research
→ Synthesize
→ Compose
→ Generate
→ Validate
```

하는 adaptive workflow여야 한다.

---

## Phase 5 — Codex Adapter

Codex 프로젝트에 생성할 `AGENTS.md` entry 구조와 Codex-specific capability를 반영.

Core artifact는 Codex에 종속시키지 않는다.

---

## Phase 6 — Validation

최소 3개 서로 다른 프로젝트 archetype으로 dry-run 권장:

```text
A. Greenfield Web App
B. Existing Legacy Bug/Refactor Project
C. Agentic AI / RAG / Experimental Project
```

확인:

- 같은 고정 Workflow가 나오지 않는가?
- Research question이 프로젝트별로 달라지는가?
- Evaluation Contract가 달라지는가?
- Handoff policy가 현실적인가?
- AGENTS.md가 비대해지지 않는가?
- Tool-specific 부분과 portable core가 분리되는가?
- future learning schema가 과도하게 현재 runtime을 복잡하게 하지 않는가?

---

# 19. Anti-Patterns — 피할 것

## 19.1 Giant AGENTS.md

모든 Workflow/Research/State를 한 파일에 넣지 말 것.

## 19.2 Fixed Universal Sequence

모든 프로젝트에:

```text
Explore → Plan → Implement → Test
```

를 무조건 강제하지 말 것.

## 19.3 Web Result Auto-Adoption

검색 결과를 검증 없이 Supervisor Rule로 채택하지 말 것.

## 19.4 Every-Turn Web Search

정보 noise / cost / method drift 증가.

## 19.5 Every-Turn MCTS

비용 폭발.

## 19.6 Self-Reported Success Trust

Coding Agent의 서술만으로 완료 판정하지 말 것.

## 19.7 Single Scalar Evaluation Only

Reward Model이 바뀌면 과거 정보 손실.

## 19.8 Generator Self-Modification from One Project

한 프로젝트 anomaly로 global behavior 오염.

## 19.9 Handoff = Full Conversation Summary

불필요한 historical noise를 다음 session에 다시 넣지 말 것.

## 19.10 Supervisor Writes Source Code by Default

Supervisor/Coder 역할 혼합으로 책임과 평가가 불명확해짐.

---

# 20. Desired Final Identity of This Skill

`Adaptive VibeCoding Supervisor Generator Skill`은 단순한 Prompt Generator가 아니다.

정확히는:

> **프로젝트의 목표·환경·위험·사용 Tool·기존 Knowledge·최신 Web Evidence를 분석하여, 해당 프로젝트에 최적화된 Vibe Coding Supervisor Harness를 생성하는 Meta-Engineering Skill.**

생성되는 Harness는:

- Goal-aware
- Evidence-driven
- Tool-adapted
- Workflow-adaptive
- Context-aware
- Session-handoff capable
- Research-enabled
- MCTS-ready
- Evaluation-instrumented
- Reward/learning-ready
- Wiki-ready
- Multi-agent-ready

해야 한다.

---

# 21. One-Sentence Design Principle

> **Adaptive VibeCoding Supervisor는 전체 미래를 고정하는 Planner가 아니라, 프로젝트 Goal과 실제 Environment Evidence를 기준으로 현재 State에서 가장 가치 있는 다음 행동을 선택하고, 결과를 관찰해 Plan/Workflow/Session을 지속적으로 조정하는 Decision Supervisor이며, Generator는 이 Supervisor를 프로젝트별로 설계·조립하는 Compiler여야 한다.**

---

# 22. Codex에게 요청하는 다음 작업

이 문서를 기반으로 바로 `SKILL.md`를 쓰기 전에 먼저 다음을 수행하라.

1. 위 요구사항을 **Must / Should / Future**로 재분류한다.
2. Generator와 생성된 Supervisor의 책임 경계를 명확히 한다.
3. `Generator / Supervisor Runtime / Planner / Evaluator / Learning / Wiki` 간 Interface Contract를 설계한다.
4. `.vibe-supervisor/` Artifact Dependency Graph를 만든다.
5. 최소 State / Decision / Reward / Evaluation / Handoff schema를 제안한다.
6. Codex Adapter의 책임과 Portable Core의 책임을 분리한다.
7. v1에서 과도한 복잡성이 있는 부분을 지적하되, future socket을 제거하지 않는다.
8. 그 후 `Adaptive VibeCoding Supervisor Generator Skill`의 구현 계획을 제안한다.
9. HUMAN 승인 후 실제 Skill 파일을 생성하는 순서로 진행한다.

**중요:** 단순히 이 문서를 Prompt로 압축하지 말고, 실제 Codex Skill의 구조적 제약과 repository 환경을 확인한 뒤 구현 가능한 형태로 변환하라.

---

# Appendix A — Conceptual Data Flow

```text
PROJECT / HUMAN
      │
      ▼
GENERATOR
      │
      ├─ Repo / Docs Inspect
      ├─ Adaptive Interview
      ├─ Project Classification
      ├─ Web Research
      ├─ Wiki Retrieval
      ├─ Evidence Synthesis
      ├─ Goal Contract
      ├─ Evaluation Plan
      └─ Workflow/Planner Policy Composition
      │
      ▼
SUPERVISOR PACKAGE
      │
      ▼
SUPERVISOR RUNTIME
      │
      ├─ Observe Coding Agent
      ├─ Read Actual Evidence
      ├─ State Reconstruction
      ├─ Drift/Risk Detection
      ├─ Planning Mode Selection
      │     ├─ Direct
      │     ├─ Simple Plan
      │     ├─ Research
      │     └─ MCTS
      ├─ Next Best Action
      ├─ Prompt/SubAgent Action
      ├─ Validation
      ├─ Reward Event
      ├─ State Update
      └─ Handoff Decision
      │
      ├──────── CONTINUE ────────┐
      │                          │
      └──────── HANDOFF          │
                │                │
                ▼                │
        Handoff Capsule          │
                │                │
                ▼                │
         New Session             │
                └────────────────┘
```

---

# Appendix B — Long-Term Learning Flow

```text
Project Runtime
     │
     ├─ Decision Trace
     ├─ Reward Events
     ├─ Session Evaluations
     └─ Final Evaluation
             │
             ▼
     Credit Assignment
     TD(λ)-inspired / future Bellman
             │
             ▼
       Pattern Extraction
             │
             ▼
          LLM Wiki
             │
             ▼
Generator Improvement Proposals
             │
       Evidence Accumulation
             │
             ▼
      HUMAN / Meta Evaluator
             │
             ▼
       Generator vNext
```

---

# Appendix C — MCTS Core Loop

```text
Current Observed State
        │
        ▼
Decision Gate
 ├─ Direct
 ├─ Simple Plan
 ├─ Research
 └─ MCTS
        │
        ▼
Candidate Actions
        │
        ▼
Limited-Horizon Simulation
        │
        ▼
Multi-dimensional Evaluation
        │
        ▼
Context-specific Value
        │
        ▼
Best Next Action
        │
        ▼
Actual Execution
        │
        ▼
Test / Build / Repo / HUMAN
        │
        ▼
Observed State
        │
        ▼
Replan
```

---

# Appendix D — Definition of Done for Generator v1

v1이 “완성”되었다고 보려면 최소:

- [ ] 서로 다른 프로젝트 유형에서 다른 Supervisor Harness를 생성한다.
- [ ] Tool Core와 Codex Adapter가 분리되어 있다.
- [ ] 프로젝트 시작 전 Goal/Evaluation Contract를 생성한다.
- [ ] Web Research가 event/TTL/evidence gate를 따른다.
- [ ] Supervisor가 Coding Agent output과 actual evidence를 구분한다.
- [ ] Workflow가 project/task/risk/uncertainty에 따라 달라진다.
- [ ] ROADMAP을 living hypothesis로 취급한다.
- [ ] CURRENT_STATE를 compact source of truth로 유지한다.
- [ ] Session Handoff를 multi-signal로 판단한다.
- [ ] Handoff + next-session bootstrap prompt를 생성할 수 있다.
- [ ] Decision/Reward/Evaluation telemetry schema가 존재한다.
- [ ] MCTS는 optional planning mode로 설계되어 있다.
- [ ] Evaluation raw vector와 planner scalarization이 분리되어 있다.
- [ ] Wiki / Learning / Multi-Agent future interface가 존재한다.
- [ ] Generator가 한 프로젝트 결과로 자기 자신을 자동 수정하지 않는다.
- [ ] Giant AGENTS.md / universal fixed workflow anti-pattern을 피한다.

