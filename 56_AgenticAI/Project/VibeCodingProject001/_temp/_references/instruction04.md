앞으로 Codex 결과를 검토할 때
Evidence를 다음 네 묶음으로 확인하라.

1. 시작 Evidence

- Root
- Branch
- Start HEAD
- Start Working tree

2. 변경 Evidence

- 변경 파일
- 파일별 변경 목적
- 범위 밖 변경 여부

3. 검증 Evidence

- 관련 테스트 결과와 실행 개수
- 전체 테스트가 필요한 경우 결과와 실행 개수
- lint
- typecheck
- build
- 실제 기능 확인
- 화면 작업이면 Runtime / Visual / Console 오류

4. 최종 Evidence

- Final HEAD
- Final Working tree
- Commit 여부
- Push 여부
- Deployment 여부
- 미해결 사항

"모두 통과했습니다."라는 표현만으로
충분한 Evidence라고 판단하지 않는다.

가능하면 실제 수치와 상태를 확인한다.

예:

52/52 PASS

Build: PASS

Console errors: 0

실행하지 않은 검증은 PASS라고 추정하지 않는다.

지금은 실제 결과를 판정하지 말고
좋은 Evidence와 부족한 Evidence의 차이를 간단히 설명하라.