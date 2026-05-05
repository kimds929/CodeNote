## Large File Handling Rule
- CSV, PDF, 로그 파일 등 대용량 파일의 원문 전체를 대화 컨텍스트에 직접 넣지 않는다.
- 대용량 데이터는 Python/도구를 이용해 스키마, 샘플, 통계 요약만 추출한다.
- 데이터 분석 초기 단계에서는 전체 원문 대신 다음만 확인한다: shape, columns, dtypes, sample rows, missing value summary, key ranges/categories
- 요약 결과는 `instructions/data_schema.md`에 저장하고, 이후 분석은 해당 문서를 우선 참조한다.