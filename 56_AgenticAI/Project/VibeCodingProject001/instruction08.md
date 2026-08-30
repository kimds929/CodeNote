앞으로 공식 PASS 커밋을 Checkpoint로 관리하라.

Checkpoint에는 최소한 다음을 기록한다.

- Gate ID
- 공식 판정
- Branch
- Start HEAD
- PASS Commit / Final HEAD
- 검증 결과 요약
- Final Working tree
- Push 여부
- Deployment 여부

다음 Gate를 시작할 때는
직전 공식 PASS Checkpoint와
현재 시작 상태의 연속성을 확인한다.

예:

Gate 02 PASS Commit = bbb2222
Gate 03 Start HEAD = bbb2222

두 상태가 다르면 바로 구현하지 말고
그 사이에 어떤 변경이 있었는지 먼저 확인한다.

공식 PASS 전에
정상 Checkpoint 커밋을 만들지 않는다.

Commit이 승인되어도
Push와 Deployment는 별도 승인으로 관리한다.

지금은 Checkpoint 연속성이
왜 필요한지 간단히 설명하라.