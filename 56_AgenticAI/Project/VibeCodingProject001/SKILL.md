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






## RULE
 - 원칙적으로 `<REPO_ROOT>/project` 폴더내의 파일만 수정한다.
 - 


