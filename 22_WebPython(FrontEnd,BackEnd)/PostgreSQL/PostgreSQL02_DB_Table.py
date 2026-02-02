############################################################################################################
# postgreSQL 시작
import json
import psycopg2

import datetime
now_date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
import pandas as pd

unlock_password = lambda pw: "".join([chr(int(pw[i:i+3])) for i in range(0, len(pw), 3)])
    
#----------------------------------------------------------------------
account_path = 'D:/DataScience/기타) 기타/'
with open(f"{account_path}/PostgreSQL_Account.json", "r") as file:
    account = json.load(file)
#----------------------------------------------------------------------
############################################################################################################


# 관리 DB(postgres)에 접속
conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    user=account['username'],
    password=unlock_password(account['password'])
)

connect_info = {'host': '127.0.0.1', 'port': 5432, 'user': account['username'], 'password': unlock_password(account['password'])}

############################################################################################################
# [DataBase 목록조희]
cur = conn.cursor() # cursor

# 데이터베이스 목록 조회
cur.execute("SELECT datname FROM pg_database;")
databases = cur.fetchall()
print([db for db in databases])  # database list

cur.close()
conn.close()

# cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
# db_list = cur.fetchall()
# print(db_list)


# ------------------------------------------------------------------------------------------------------------------------------
# [DataBase 생성/삭제]
def db_list(cur):
    cur.execute("SELECT datname FROM pg_database;")
    databases = cur.fetchall()
    return [db for db in databases]  # database list

# connection
conn = psycopg2.connect(**connect_info)
cur = conn.cursor() # cursor

conn.autocommit = True  # CREATE/DROP DATABASE는 autocommit 필요
print( db_list(cur) )   # DB_list확인

db_name = 'test_db'
# DB 생성
# cur.execute(f"CREATE DATABASE {db_name};")
print( db_list(cur) )   # DB_list확인

# DB 삭제
# cur.execute(f"DROP DATABASE {db_name};")      # DB 안의 모든 테이블과 데이터가 함께 삭제
print( db_list(cur) )   # DB_list확인

cur.close()
conn.close()



############################################################################################################
# [DataBase 접근 및 Table 목록조희]
conn = psycopg2.connect(**connect_info, database='postgres')
cur = conn.cursor() # cursor

cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name;
""")
tables = cur.fetchall()
print([tb for tb in tables])  # table list

cur.close()
conn.close()





############################################################################################################
# 테이블 생성 / 삭제 / 수정 / 조회
def tb_list(cur):
    cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name;
    """)
    tables = cur.fetchall()
    return [tb for tb in tables]  # table list

# 새로 만든 DB에 접속
conn = psycopg2.connect(**connect_info, database='postgres')
cur = conn.cursor()     # cursor
print( tb_list(cur) )     # tb_list확인

# 테이블 생성
cur.execute("""
CREATE TABLE test_table (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50),
    hire_date DATE
);
""")
print( tb_list(cur) )     # tb_list확인
# [Date Type]
# (정수형)
#     SMALLINT (2바이트, -32,768 ~ 32,767)
#     INTEGER 또는 INT (4바이트, 약 ±21억)
#     BIGINT (8바이트, 매우 큰 정수)
# (자동증가 정수)
#     SERIAL (INTEGER + 자동 증가)
#     BIGSERIAL (BIGINT + 자동 증가)
# (실수형)
#     REAL (4바이트, 부동소수점)
#     DOUBLE PRECISION (8바이트, 부동소수점)
#     FLOAT(n)    (가변) n 값에 따라 REAL 또는 DOUBLE PRECISION으로 자동 매핑 (FLOAT(10) → REAL, FLOAT(50) → DOUBLE PRECISION)
# (고정 소수점)
#     NUMERIC(precision, scale) 또는 DECIMAL (정밀한 소수 계산 가능)
# (문자형)
#     CHAR(n) : 고정 길이 문자열
#     VARCHAR(n) : 가변 길이 문자열 (최대 n자)
#     TEXT : 길이 제한 없는 문자열
# (날짜/시간형)
#     DATE : 날짜 (YYYY-MM-DD)
#     TIME : 시각 (HH:MM:SS)
#     TIMESTAMP : 날짜 + 시각
#     TIMESTAMPTZ : 타임존 포함 날짜+시각
#     INTERVAL : 기간(예: '2 days', '3 hours')
# (불리언)
#     BOOLEAN : TRUE / FALSE
# (기타)
#     BYTEA : 이진 데이터(바이너리)
#     UUID : 범용 고유 식별자
#     JSON, JSONB : JSON 데이터 저장
#     ARRAY : 배열 타입   (지정방법) TEXT[], INTEGER[], NUMERIC(10,2)[]
#     ENUM : 열거형
#     GEOMETRY / GEOGRAPHY : 공간 데이터(PostGIS 확장 필요)


# 데이터 입력
cur.execute("""
INSERT INTO test_table (name, department, hire_date)
VALUES (%s, %s, %s);
""", ("홍길동", "R&D", "2024-06-01"))

# 데이터 수정
cur.execute("""
UPDATE test_table
SET department = %s
WHERE name = %s;
""", ("기술연구소", "홍길동"))

# 데이터 조회
cur.execute("SELECT * FROM test_table;")
rows = cur.fetchall()
columns = [desc.name for desc in cur.description]
pd.DataFrame(rows, columns=columns)

# 테이블 삭제
cur.execute("DROP TABLE test_table;")
print( tb_list(cur) )     # tb_list확인

conn.commit()
cur.close()
conn.close()


# ------------------------------------------------------------------------------------------------------------------------------
# [TABLE KEY]
#     PRIMARY KEY	 : 테이블당 1개	(행 고유 식별)	무결성보장 O	자동인덱스생성 O
#     FOREIGN KEY	 : 여러 개 가능	(다른 테이블 참조)	무결성보장 O	자동인덱스생성 X
#     INDEX(Secondary Key)	: 여러 개 가능	(검색 속도 향상)	무결성보장 X	자동인덱스생성 O(검색용)

from datetime import date

def retrieval_all_data(cur, script):
    cur.execute(script)
    rows = cur.fetchall()
    columns = [desc.name for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)

conn = psycopg2.connect(**connect_info, database='postgres')

# 커서 생성
cur = conn.cursor()

# SQL 스크립트 --------------------------------------------------------------------------------------------------------
sql_script = """
-- 부서 테이블
CREATE TABLE IF NOT EXISTS department (
    dept_id SERIAL PRIMARY KEY,          -- PRIMARY KEY: 부서 고유 ID
    dept_name VARCHAR(50) NOT NULL UNIQUE -- UNIQUE: 부서 이름은 중복 불가
);

-- 직원 테이블
CREATE TABLE IF NOT EXISTS employee (
    emp_id SERIAL PRIMARY KEY,           -- PRIMARY KEY: 직원 고유 ID
    name VARCHAR(100) NOT NULL,           -- 직원 이름
    department_id INT NOT NULL,           -- 부서 ID (외래키)
    hire_date DATE NOT NULL,              -- 입사일
    salary NUMERIC(10, 2),                 -- 급여

    -- FOREIGN KEY: department 테이블의 dept_id를 참조
    FOREIGN KEY (department_id) REFERENCES department(dept_id)
);

-- INDEX(Secondary Key): 직원 이름으로 검색 속도 향상
CREATE INDEX IF NOT EXISTS idx_employee_name ON employee(name);

-- INDEX: 입사일로 검색 속도 향상
CREATE INDEX IF NOT EXISTS idx_employee_hire_date ON employee(hire_date);
"""

# SQL 실행 
cur.execute(sql_script)

# 변경사항 저장
conn.commit()

print("테이블과 인덱스가 성공적으로 생성되었습니다.")
print( tb_list(cur) )     # tb_list확인

# 2. 샘플 데이터 추가 --------------------------------------------------------------------------------------------------------
# departments
departments = [
    ("철강사업부",),
    ("에너지사업부",),
    ("연구개발부",)
]
cur.executemany("INSERT INTO department (dept_name) VALUES (%s) ON CONFLICT (dept_name) DO NOTHING;", departments)

# employees
employees = [
    ("김철수", 1, date(2020, 3, 15), 50000000),
    ("이영희", 2, date(2019, 7, 1), 60000000),
    ("박민수", 1, date(2021, 1, 10), 45000000),
    ("최지현", 3, date(2018, 11, 20), 70000000)
]
cur.executemany(
    "INSERT INTO employee (name, department_id, hire_date, salary) VALUES (%s, %s, %s, %s);",
    employees
)

conn.commit()
print("샘플 데이터 삽입 완료.")


# 데이터 조회 (retreival data) --------------------------------------------------------------------------------------------------------
retrieval_all_data(cur, "SELECT * FROM department;")    # 데이터 조회
retrieval_all_data(cur, "SELECT * FROM employee;")    # 데이터 조회

join_script = """
SELECT e.emp_id, e.name, d.dept_name, e.hire_date, e.salary
FROM employee e
JOIN department d ON e.department_id = d.dept_id
ORDER BY e.emp_id;
"""
retrieval_all_data(cur, join_script)


# 테이블 삭제 --------------------------------------------------------------------------------------------------------
cur.execute("DROP TABLE department CASCADE;")   # CASECADE : FOREIGN KEY 의존성까지 고려해서 삭제
cur.execute("DROP TABLE employee CASCADE;")
conn.commit()

print( tb_list(cur) )     # tb_list확인


# 연결 종료
cur.close()
conn.close()