---
name: project_supervisor
description: 프로젝트의 목표·환경·위험·사용 도구를 분석해, 해당 프로젝트의 Coding Agent를 감독할 Supervisor workflow와 Package를 설계·생성·검증한다. 직접 개발보다 프로젝트별 감독체계가 필요할 때 사용한다.

---

# 스킬: project_supervisor

## 목적
이 스킬은 코드를 대신 만드는 도구가 아니라, **프로젝트별로 “어떻게 개발을 감독할지”를 설계하는 도구**다.
프로젝트 정보, Human의 목표·제약, 사용 가능한 도구를 바탕으로 위험도에 맞는 질문·조사·검증 기준·실행 통제를 구성한다.
그 결과 Coding Agent의 작업을 지시하고 Evidence로 검증하며 상태와 handoff를 관리하는 Supervisor Harness/Package를 생성·검증한다.

## 정의
### 한 문장 정의

`project_supervisor`는 **프로젝트를 분석해 그 프로젝트에 맞는 VibeCoding Supervisor를 설계·조립하는 Meta-Engineering Skill**이다.

### 무엇을 받아서 무엇을 만드는가

```text
입력
= 프로젝트 저장소·문서
  + Human의 목표·성공 기준·제약
  + 사용 가능한 Agent·Tool·검증 환경

판단
= 프로젝트 유형·위험·복잡도·불확실성·가역성 평가
  → 필요한 Supervisor 기능과 통제 수준 선택

출력
= Goal / Evaluation / Authority 계약
  + 프로젝트 맞춤 Workflow·Evidence·Recovery·Handoff 정책
  + 실행 가능한 Supervisor Harness/Package
```

### 역할 경계

| 구성요소 | 하는 일 |
|---|---|
| Generator Skill | 프로젝트를 이해하고 Supervisor Package를 설계·생성·검증한다. |
| Supervisor Runtime | 생성된 Package를 사용해 현재 상태를 관찰하고 다음 작업·검증·handoff를 관리한다. |
| Coding Agent | Supervisor의 지시에 따라 실제 소스 코드를 수정하고 실행·테스트한다. |

### 핵심 용어

- `<REPO_ROOT>`: 감독 대상 프로젝트의 최상위 경로.
- `<SKILL_ROOT>`: 이 `SKILL.md`가 들어 있는 `project_supervisor` 스킬 폴더의 경로.
- <PROJECT_ROOT>: Supervisor가 생성·관리하는 프로젝트 운영 폴더. 기본값은 <REPO_ROOT>/project/.
- `Supervisor Package/Harness`: 프로젝트 목표, 상태, Workflow, 평가 기준, 권한, Evidence, Recovery, handoff 규칙을 담은 감독 운영체계.
- `Evidence`: 테스트 출력, Git 상태, 실행 결과처럼 실제 상태를 확인할 수 있는 관찰 근거.


## 절차
### 1. 프로젝트 진행상태 파악
 - <PROJECT_ROOT> 존재 여부 확인
 - <PROJECT_ROOT>/AGENTS.md 또는 Supervisor 상태 파일 존재 여부 확인
 - 기존 Repository의 코드/문서/변경 이력 존재 여부 확인

판단:
 - Supervisor 운영정보 없음 → 신규 Supervisor 초기화
 - Supervisor 운영정보 있음 → 기존 프로젝트/세션 재개
 - 상태가 불분명하거나 서로 충돌함 → Repository 상태를 추가 확인 후 결정

### 2. 신규프로젝트 진행
#### 2.1. Project Discovery
##### 2.1.1. Capability & Environment Discovery
 - Repo 접근 가능 여부
 - Git 사용 가능 여부
 - Shell / terminal 실행 가능 여부
 - Browser 사용 가능 여부
 - MCP / 외부 Tool 사용 가능 여부
 - Subagent 사용 가능 여부
 - Web Research 가능 여부
 - Artifact 작성 가능 위치

##### 2.1.2. Repository & Project Baseline Discovery
 - <REPO_ROOT> 절대 경로
 - 주요 디렉터리 구조
 - 기존 코드 존재 여부
 - AGENTS.md / README / docs 존재 여부
 - package manifest
 - Git branch / HEAD / working tree
 - 최근 변경사항
 - test / lint / build 명령
 - CI/CD

##### 2.1.3. Runtime & Validation Environment Discovery
 - Language / Runtime
 - Python / Node / Java 등의 버전
 - Interpreter / executable 경로
 - Browser validation
 - Visual validation
 - Test environment
 - Database / API 접근 가능 여부
 - Local / staging / production 구분

#### 2.2 Project Interview
##### Interview 원칙
 - 프로젝트 인터뷰는 질의응답식으로 진행된다.
 - 첫 질문은 고정된 질문으로 시작하여, 그 답변 결과에 따라 가장 중요한 부족 정보를 물어보는 방식으로 연쇄적으로 질문이 이어진다.
 - 질문은 전문용어 사용을 지양하고(필요시 빈칸에 표기), 쉬운 용어를 사용하거나 사용자의 목적을 질문하여 그에 맞는 선택지를 Agent가 골라준다.
 - 꼭 필요한 질문만 하며, 가능한 범위내에서 Interview 횟수를 최소화한다.
 - 선택지 Option
    - 선택지가 자연스럽게 존재하는 질문에서는 실제 의사결정 가능한 Option을 우선 제공한다.
    - 대안 탐색이 필요한 질문에서는 가능하면 다음 3개 수준을 포함한다.
        1. 현재 프로젝트에 가장 적합한 현실적 선택
        2. 품질 또는 확장성을 높이는 선택
        3. 범위를 확장하는 탐색적 선택
    - 추천안에는 `(추천)`을 표시하고 한 줄 이유를 함께 제공한다.
    - 반드시 각 선택지마다 장점, 단점을 매우 간략히 설명한다.
 - Repository, 기존 문서, 설정 파일, 실행 환경 또는 사용 가능한 Tool로 직접 확인 가능한 정보는 Human에게 다시 질문하지 않는다.
 - 반드시 해야하는 질문 Category : 목표, 결과물의 형태, 사용가능한 Source, 제약 조건
 - 필요시 질문할 수 있는 Category : 배경, 대상 사용자, 성공기준, 검증방법, 벤치마크 여부, 장기 확장방향
 - Agent는 내부적으로 아래 상태를 갱신하되, Human에게는 새롭게 확정된 사항과 현재 중요한 미확정 사항만 간결하게 보여준다.
    1. 확인된 사실 (confirmed, agent-resolved)
    2. Human의 목표와 기대결과
    3. 추정한 내용
    4. 아직 확인되지 않은 내용 (hold)
    5. 확인되지 않은 내용중에 현재 중요도 기반으로 Agent가 자체 판단 가능한 내용
    6. 반드시 Human의 추가 질문이 필요한 내용

#### Interview 절차
##### STEP 1. 첫 질문(서술식) : 진행하고자 하는 프로젝트에 대해 최대한 자세히 써주세요. (배경, 목표 및 결과물, 사용가능한 Source, 제약조건 등) 

##### STEP 2. 두번째 질문(Option) : Project 수행과정에서 Human개입을 얼마나 원하는가?
    Option 1. 최소 : Agent가 안전한 범위에서 스스로 판단하고, 중요 의사결정이나 Gate 실패 시에만 Human에게 묻는다.
    Option 2. 중간 : Phase/Gate 단위로 진행 결과와 다음 계획을 Human에게 확인받는다.
    Option 3. 높음 : 주요 설계·구현 방향을 결정하기 전에 Human 승인을 받는다.

##### STEP 3. 첫 질문을 바탕으로 Project의 복잡도와 중요도를 평가한다.
    - 복잡도 : 구현·판단·검증이 얼마나 어려운가 (Agent 자체판단)
        1. 낮음 : 한두 개 파일 또는 독립된 기능. 영향 범위가 작음. 테스트와 되돌리기가 쉬움.
        2. 중간 : 여러 파일·모듈 또는 시스템이 연결됨. 반드시 테스트가 필요. 테스트에 추가적인 확인이 필요함.
        3. 높음 : 여러 시스템 연동. 영향 범위가 넓거나 검증이 어려움. 테스트시 다각도의 검증이 필요.
    - 중요도 : 실패했을 때 얼마나 큰 영향을 받는가 (default : 낮음)
        1. 낮음 : 실패해도 사용자·데이터·서비스에 큰 영향이 없음. 쉽게 원상복구하거나 다시 만들 수 있음.
        2. 중간 : 여러 파일·기능·사용자에게 영향을 줌. 복구는 가능하지만 시간과 비용이 필요함.
        3. 높음 : 고객 서비스, 금전, 보안, 운영시스템과 관련. 실패를 쉽게 되돌리기 어려움.

##### STEP 4. Interview를 완료하기 위한 Gate를 설계한다.
Interview 완료판단을 위한 Gate의 종류는 Agent가 첫질문 답변에 기반하여 설정한다.
Gate 상태는 아래 2가지로 나뉜다. 
1. confirmed : Human 또는 Evidence로 명확하게 확인됨
2. agent-resolved : Human 확인은 없지만 Agent가 Human개입정도를 고려하여 합리적으로 결정함
3. hold : 확인되지 않은 내용(불확실성 불충족), 핵심정보가 없거나 답변이 서로 충돌함.

##### STEP 5. 모든 Gate가 confirmed/agent-resolved 될 때까지 Interview를 이어간다. (Loop)
Interview 과정에서 복잡도나 중요도를 재평가하고 그에 따른 Gate 종류도 변경될 수 있다.
기존에 confirmed 또는 agent-resolved 된 Gate라도 새로운 Evidence나 Human의 답변과 충돌하는 경우 다시 hold로 되돌릴 수 있다.


##### STEP 6. Interview 종료
모든 Gate 상태가 confirmed/agent-resolved인 경우 Interview 결과를 구조화해서 Human에게 보여준다. 
그 내용에는 아래 사항이 포함된다.
- Interview Summary
- Interview 과정에서 결정된 내용
    2.1. Human이 결정한 사항에는 마지막에 `(Human)`을 붙인다.
    2.2. Agent가 결정한 사항에는 마지막에 `(Agent)`를 붙인다.
마지막에 후에  묻는다 : 이대로 Project정의를 확정할까요?
    Option 1. 확정한다. --> `AGENTS.md` 생성
    Option 2. 예상되는 프로젝트 결과물을 보고 결정한다 --> 대략적인 결과물 형태를 Project 유형에 적합한 Preview Artifact(가능하다면 HTML)로 생성하여 Human에게 보여준다. --> 다시 마지막 질문을 이어서한다.
    Option 3. 수정/보완이 필요한 사항을 적어주세요. --> Human의 prompt에 따라 의견을 반영하거나 Interview를 추가로 진행한다.
    * Preview Artifact는 최종 산출물 자체를 구현하는 것이 아니라,Human이 Project Definition과 예상 결과 형태를 검토하기 위한 최소 수준의 표현물이어야 한다.


#### 2.3. Project Classification & Control Profile

Discovery와 Interview 결과를 기반으로 프로젝트 특성과 필요한 감독 수준을 결정한다.

평가 요소:
- Project Type
- Complexity
- Importance / Impact
- Uncertainty
- Reversibility
- Verification Feasibility
- Human Intervention Preference
- Available Capability

이를 바탕으로 `Control Profile`을 결정한다.

Control Profile은 이후 다음 사항의 선택 기준으로 사용한다.
- Workflow의 깊이
- Gate 및 Evidence 강도
- Human 승인 범위
- Research 필요 수준
- Subagent 검증 필요 여부
- Recovery / Handoff 수준
- Supervisor Package 구성


#### 2.4. Information Sufficiency Gate

Supervisor Package를 설계하기에 정보가 충분한지 판단한다.

핵심 정보가 부족한 경우:
- Human만 결정 가능 → Interview로 돌아간다.
- Repository / Tool로 확인 가능 → 추가 Discovery를 수행한다.
- 외부 지식이 필요 → Conditional Research를 수행한다.
- 낮은 위험이며 합리적으로 결정 가능 → Agent가 결정하고 기록한다.

Supervisor 설계에 영향을 주는 핵심 `hold`가 없을 때 다음 단계로 진행한다.


#### 2.5. Conditional Research

현재 정보만으로 적절한 Supervisor Workflow 또는 검증 방법을 신뢰성 있게 결정하기 어려운 경우에만 Research를 수행한다.

대표 Trigger:
- 새로운 기술 / Framework / Tool
- Architecture 선택
- 보안 / DB / Deployment 등 고위험 결정
- 최신 정보 확인 필요
- 반복 실패 또는 낮은 Confidence

Research 결과는 Project에 적합한지 평가한 뒤 채택하며,
채택된 내용과 주요 근거를 Supervisor Package에 반영한다.


#### 2.6. Supervisor Package Design

Project Definition과 Control Profile을 기반으로
해당 프로젝트에 필요한 Supervisor Package의 구성과 내용을 설계한다.

모든 프로젝트에 동일한 폴더구조 및 Package를 생성하지 않고, 프로젝트 특성에 따라 결정한다.
프로젝트 특성·위험·복잡도·불확실성·Human 개입 수준 및 사용 가능한 Tool에 따라
필요한 Directory와 Supervisor 기능, Artifact를 선택·구성한다.


##### 2.6.1. Adaptive Project Workspace Design

Supervisor 운영을 위한 전용 `<PROJECT_ROOT>`를
기본적으로 `<REPO_ROOT>/project/`에 구성한다.

`<PROJECT_ROOT>`는 프로젝트의 계획, 상태, Workflow, Evidence,
Research, Decision, Handoff 등 Supervisor가 생성·관리하는
운영 Artifact의 기본 저장 영역이다.

Project Definition과 Control Profile을 기반으로
먼저 `<PROJECT_ROOT>` 내부의 Project-specific Directory Structure를 설계한다.

Directory Structure는 다음 요소를 고려하여 Adaptively 결정한다.

- Project Type 및 Domain
- Project 규모와 예상 기간
- Complexity / Risk / Uncertainty / Reversibility
- 필요한 Workflow와 Gate의 수
- Research 필요 여부
- Evidence 및 Evaluation 방식
- Human 승인 및 Decision 기록 필요성
- Session / Handoff 필요성
- 사용 가능한 Agent / Tool / Validation Capability

불필요한 Directory나 Artifact는 생성하지 않는다.

##### 2.6.2. Package Component Selection

기본적으로 최소한의 Core Artifact만 생성하고,
Project Control Profile에 따라 필요한 Artifact를 추가한다.

예:

- 단순·저위험 Project
  → AGENTS.md + PROJECT.md + ROADMAP.md + CURRENT_STATE.md

- 검증이 중요한 Project
  → Core + GATES.md + EVALUATION.md

- Human 승인 경계가 중요한 Project
  → Core + AUTHORITY.md

- 높은 불확실성 / 최신 기술 사용
  → Core + RESEARCH.md

- 장기·다단계 Project
  → Core + GATES.md + DECISIONS.md + HANDOFF.md

- 복잡·고위험 Project
  → 필요한 모든 감독 Artifact를 조합한다.

필요하지 않은 Artifact는 생성하지 않는다.


##### 2.6.3. Package Content Composition

선택된 Artifact의 내용은 Project Definition과 Control Profile을 기반으로 생성한다.

- `AGENTS.md`
  → Coding Agent가 반드시 따라야 할 핵심 감독 규칙과 읽어야 할 Project Artifact의 Entry Point

- `PROJECT.md`
  → Goal, Scope, Non-goal, Constraint, Success Definition

- `ROADMAP.md`
  → 현재 시점의 best-known Phase / Gate / Milestone

- `CURRENT_STATE.md`
  → 현재 Phase, Active Task, 상태, Evidence, Issue, Next Action

- `GATES.md`
  → Gate별 목표, 통과 기준, Evidence, Recovery

- `EVALUATION.md`
  → Project 성공 및 작업 결과 평가 방법

- `AUTHORITY.md`
  → Agent 자동 판단 범위와 Human 승인 범위

- `RESEARCH.md`
  → 채택한 외부 근거와 Project 적용 결정

- `DECISIONS.md`
  → 중요한 Human / Agent 결정과 변경 이유

- `HANDOFF.md`
  → 다음 Session에서 상태를 복구하기 위한 최소 정보


##### 2.6.4. Supervisor Runtime Contract

생성된 Supervisor는 기본적으로 다음 Loop를 따른다.

1. Goal과 Current State를 확인한다.
2. 실제 Repository / Evidence와 기록된 상태를 비교한다.
3. 현재 Gate와 다음 작업을 결정한다.
4. Coding Agent에게 작업 범위와 요구 Evidence를 지시한다.
5. 결과를 Evidence로 검증하여 PASS / FAIL / HOLD를 판단한다.
6. State / Roadmap / Handoff를 갱신하고 다음 행동을 결정한다.

Project Control Profile에 따라 각 단계의 통제 강도를 조절한다.


#### 2.7. Supervisor Package Generation

설계된 Package Profile에 따라 `<PROJECT_ROOT>`에 필요한 Artifact를 생성한다.

생성 원칙:
- 필요한 파일만 생성한다.
- 중복 정보를 최소화한다.
- Stable 정보와 Current State를 분리한다.
- AGENTS.md는 Entry Point로 유지한다.
- 각 Artifact의 책임을 겹치지 않게 한다.
- Human / Agent 결정사항의 출처를 보존한다.

생성 후 Initial Roadmap과 Current State를 초기화한다.



#### 2.8. Package Validation / Dry Run

생성된 Supervisor Package가 실제 프로젝트를 감독할 수 있는지
소스 코드를 수정하지 않고 검증한다.

검증 항목:
- Human의 Goal이 정확히 반영되었는가
- Workflow가 Project 특성과 Risk에 적합한가
- 각 중요한 Gate에 판단 가능한 Evidence가 있는가
- Human 승인 범위가 명확한가
- 실패 시 Recovery 경로가 있는가
- Package Artifact 간 상태가 모순되지 않는가
- 다음 Session에서 Current State만으로 작업을 복구할 수 있는가

검증 실패 시 관련 설계 단계로 돌아가 수정 후 다시 검증한다.


#### 2.9. Human Finalization

검증 결과와 생성된 Supervisor Package의 핵심 내용을 Human에게 보여준다.

Human에게 최소 다음을 확인한다.
- Project Definition
- 주요 Workflow / Gate
- Human 승인 범위
- Supervisor 통제 수준
- 생성된 Artifact
- Agent가 자체 결정한 중요 사항

승인:
→ Supervisor Package 활성화

수정:
→ 관련 단계로 돌아가 수정 후 다시 검증

핵심 불확실성 발견:
→ Interview 또는 Information Sufficiency Gate로 돌아간다.



### 3. 기존 Supervisor 프로젝트 재개

기존 Supervisor Package가 존재하면 새로 초기화하지 않는다.

1. AGENTS.md와 CURRENT_STATE.md를 우선 확인한다.
2. 실제 Repository / Git 상태와 기록된 상태를 비교한다.
3. 기존 Roadmap, Gate, 미해결 Issue를 복구한다.
4. 불일치가 없으면 현재 작업부터 재개한다.
5. 중요한 충돌이 있으면 관련 Gate를 hold로 변경하고 원인을 확인한다.

기존 Human 결정과 Project History를 보존하며 필요한 정보만 갱신한다.



### 4. 공통 운영 원칙
 - 원칙적으로 `<REPO_ROOT>/project` 폴더내의 파일만 수정한다.
 - `AGENTS.md`는 Supervisor Package 전체 내용을 중복해서 담지 않는다.
 - Coding Agent가 현재 작업을 수행하기 위해어떤 Artifact를 어떤 순서로 확인해야 하는지 안내하는 Entry Point 역할을 한다.
 - 필요한 정보만 단계적으로 읽도록 하여 불필요한 Context 증가를 방지한다.
- Human의 명시적 Goal과 결정은 Agent의 추정보다 우선한다.
- Agent의 주장보다 Repository, Test, Runtime 등 실제 Evidence를 우선한다.
- 중요한 Gate는 Evidence 없이 PASS 처리하지 않는다.
- Risk가 높고 가역성이 낮을수록 Human 승인과 검증을 강화한다.
- 새로운 Evidence가 기존 판단과 충돌하면 이미 통과한 Gate도 재검토한다.
- 불필요한 질문·Artifact·절차를 만들지 않는다.
- Supervisor는 직접 Coding하는 Agent가 아니라 방향·상태·검증을 관리하는 감독자다.

