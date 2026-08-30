앞으로 Codex의 결과 보고와
공식 Gate 판정을 구분하라.

Codex가 모든 작업을 완료했다고 판단하면
결과 상태를 PASS_CANDIDATE로 보고할 수 있다.

그러나 PASS_CANDIDATE는 공식 PASS가 아니다.

공식 판정은 Codex가 제출한 Evidence와
현재 Gate의 기준을 검토한 뒤 결정한다.

[PASS]

다음 조건이 충족된 상태이다.

- Gate 목적 달성
- 필수 검증 완료
- 보호 범위 유지
- 필요한 Evidence 충분
- 다음 Gate 진행을 막는 문제가 없음

[FAIL]

다음 중 필수 기준에 문제가 있는 상태이다.

- 기능
- 테스트
- 검증
- 보호 범위

[HOLD]

다음과 같은 이유로 실행 또는 판정을 완료할 수 없는 상태이다.

- 정보 부족
- 환경 부족
- 권한 부족
- 안전조건 미충족

일부 성공,
Build 성공,
Codex의 "작업 완료" 문구만으로
공식 PASS를 선언하지 않는다.

앞으로 내가 Codex 결과를 전달하면

1. Evidence를 먼저 요약하고
2. 그 다음 공식 PASS / FAIL / HOLD를 판정하라.

지금은 PASS_CANDIDATE와 공식 PASS의 차이를 설명하라.