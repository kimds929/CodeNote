---
name: wiki-project-setting
description: 내 프로필과 AGENTS.md를 바탕으로 LLM Wiki 구조(지식 습득 + 실행 레이어)를 승인 기반으로 세팅하는 스킬. "wiki-project-setting 스킬로 내 Vault를 세팅해줘"로 실행.
---

# 스킬: wiki-project-setting

## 목적

나만의 LLM Wiki 뼈대를 만든다 — 구조는 AI가 제안하고, 승인은 사용자가 한다.

## 입력
- `내 프로필.md` · `AGENTS.md` (없으면 사용자에게 하는 일·지식 주제·일의 단위를 짧게 질문)

## 절차

1. 내 프로필과 AGENTS.md를 읽는다.
2. **구조안을 먼저 제안**한다 — 항목마다 프로필 근거 한 줄:
   - `raw/` — 인풋 유형별 하위 폴더 (원본 보관, 불변)
   - `wiki/` — 지식 주제를 카테고리로
   - `output/` — 산출물
   - 실행 폴더 — 일의 단위대로 (`Meeting` · `Projects` 등)
   - **모두 Vault 최상위에 — 실행 폴더는 wiki 안이 아니라 wiki 옆에**
3. 사용자가 승인하면 생성한다: 폴더 + `wiki/INDEX.md`(카테고리별 한 줄) + `wiki/log.md`
4. AGENTS.md에 **위키 운영 규칙**을 추가한다:
   1. raw는 수정 금지 — 불변 원본
   2. wiki 노트를 만들거나 고치면 INDEX.md 갱신
   3. 모든 위키 작업은 log.md에 한 줄 기록
   4. 내부 참조는 [[위키링크]]
   5. 모든 wiki 노트에 YAML frontmatter
   6. 질문에는 INDEX부터 — raw는 마지막 수단
   7. 새 페이지보다 기존 페이지 업데이트 우선
   8. 요약은 사실만 — 해석은 구분해서
   9. 출처가 모순되면 양쪽 다 인용
   10. INDEX 항목은 한 줄로

5. **스킬 자가 설치**: `skills/` 폴더(global 및 현재 local workspace 모두 확인) 에 `wiki-ingest` 그리고 `wiki-query` 스킬이 없으면 함께 만들어준다 (이미 있으면 건너뛴다):
   - 스킬 만들기 전 반드시 해당 Skill을 Global로 등록할 것인지? Local로 등록할 것인지 의견을 확인 후에 생성한다.
   - 생성시 참고사항 : 스킬 생성시 아래 link의 내용을 참고해서 그 사상이 반영될 수 있도록 구현한다.
      1. OKF 사상 : https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md
      2. LLM-wiki 사상 : https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
   - `skills/wiki-ingest/SKILL.md` — raw의 원본에서 시간이 지나도 쓸 지식만 뽑아 wiki 노트로 컴파일 (원본 불변 · 기존 노트 우선 업데이트 · frontmatter+출처 · INDEX·log 갱신)
   - `skills/query/SKILL.md` — INDEX부터 찾아 출처 위키링크와 함께 답하고, 없으면 지어내지 않고 `wiki-ingest`를 제안

## 출력

- 폴더 구조 · `wiki/INDEX.md` · `wiki/log.md` · AGENTS.md 운영 규칙 · (없을 때) `wiki-ingest`·`wiki-query` 스킬

## 규칙

- **승인 전에는 아무것도 생성하지 않는다.**
- 프로필에 없는 내용은 지어내지 말고 물어본다.
