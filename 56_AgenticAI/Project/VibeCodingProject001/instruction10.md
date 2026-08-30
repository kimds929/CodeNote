지금까지 학습한 GateOps Level 2 운영 기준을 최종 확인한다.

아래 항목에 각각

"이해"
또는
"재확인 필요"

로 답하고 한 줄씩 이유를 설명하라.

1. Baseline을 확인한 뒤 Gate를 시작한다.

2. Gate에는 진입조건과 보호 범위를 명확하게 둔다.

3. Codex 지시문에는 중단조건과 Evidence 요구를 포함한다.

4. Codex의 PASS_CANDIDATE와 공식 PASS를 구분한다.

5. FAIL이면 다음 정상 Gate 대신 Recovery Gate를 설계한다.

6. 범위 밖 문제는 자동 수정하지 않고 별도 후보로 기록한다.

7. PASS Checkpoint와 다음 Gate의 Start HEAD 연속성을 확인한다.

8. Commit / Push / Deployment를 별도 승인으로 관리한다.

9. 위험도가 높은 작업은 조사·구현·검증을 필요에 따라 분리한다.

10. Evidence가 부족하면 추정으로 PASS하지 않는다.

모든 항목을 이해했다면
마지막 줄에 다음과 같이 답하라.

GateOps Level 2 운영 준비 완료

이후 실제 프로젝트 요청에서는
위 Level 2 운영 기준을 적용하라.