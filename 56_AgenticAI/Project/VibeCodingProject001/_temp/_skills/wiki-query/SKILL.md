---
name: wiki-query
description: 쌓인 wiki에 질문하고 출처와 함께 답하는 스킬. 좋은 답은 다시 wiki에 저장한다 (Knowledge Flywheel). "wiki-query 스킬로 ○○를 찾아줘"로 실행.
---

# 스킬: query

wiki 지식 기반에 질문을 던지고, 이미 정리된 노트를 기반으로 답하는 워크플로우. 매번 원본을 뒤지는 게 아니라 — **wiki가 1차 소스, raw는 마지막 수단.**

## 절차

1. **INDEX 먼저** — `wiki/INDEX.md`를 읽고 질문과 관련된 노트를 찾는다.
2. 관련 wiki 노트들을 읽고 종합한다 — **모순이 있으면 양쪽 다 인용**한다.
3. wiki만으로 부족할 때만 `raw/`를 본다. raw에서 답을 찾았다면 그건 wiki에 구멍이 있다는 신호 — `wiki-ingest`를 제안한다.
4. 답변한다 — 모든 핵심 주장에 출처를 단다: `출처: [[노트 이름]]`
5. **Knowledge Flywheel**: 답변이 새로운 연결·통합이라 가치 있으면 wiki 저장을 제안한다 — 저장 시 INDEX 갱신 + `wiki/log.md`에 기록:
   `- [YYYY-MM-DD] wiki-query | 질문 요약`

## 출력

- 출처 달린 답변 · (가치 있으면) wiki에 저장된 새 노트

## 규칙

- wiki에 없는 내용을 지어내지 않는다 — "wiki에 아직 없습니다, ○○를 wiki-ingest 할까요?"라고 말한다.
