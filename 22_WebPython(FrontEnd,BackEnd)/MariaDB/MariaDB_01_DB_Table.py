############################################################################################################
# postgreSQL 시작
import pymysql
from pymysql.cursors import DictCursor
from datetime import datetime, timedelta
import json

now_date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
import pandas as pd

# -------------------------------------------------------------------------------------------------------------------
def check_refresh_password(account, DB='PKNMAS', verbose=1):
    refresh_periods = {'PKNMAS':20, 'PDLADB':50}
    last_update = datetime.strptime(account['last_update'], "%Y%m%d")
    now = datetime.now()
    update_deadline = last_update + timedelta(days=refresh_periods[DB])
    
    remain_days = (update_deadline - now).total_seconds() /(60*60*24)
    if remain_days < 0:
        print(f"** ExpiredError:  ORA-28001: the password has expired.(deadline{datetime.strftime(update_deadline, '%Y-%m-%d')})")
    elif remain_days < 5:
        print(f"* ExpiredWarning: password refresh period is about to expire. ({int(remain_days)} days remain, deadline{datetime.strftime(update_deadline, '%Y-%m-%d')})")
    elif verbose > 0:
        print(f"NoticeRefreshDeadline : {datetime.strftime(update_deadline, '%Y-%m-%d')}")
    

def refresh_password(account):
    account['last_update'] = datetime.now().strftime("%Y%m%d")
    with open(f"{account_path}/MariaDB_Account.json", "w", encoding='utf-8') as file:
        json.dump(account, file)
# -------------------------------------------------------------------------------------------------------------------


#----------------------------------------------------------------------
account_path =  'D:/DataScience/PythonForwork/MariaDB'
with open(f"{account_path}/MariaDB_Account.json", "r") as file:
    account = json.load(file)
#----------------------------------------------------------------------
check_refresh_password(account)

############################################################################################################

# MySQL 접속
connect_info = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': account['username'],
    'password': account['password']
}

conn = pymysql.connect(**connect_info)
cur = conn.cursor()


############################################################################################################
# [DataBase 목록조희]
cur.execute("SHOW DATABASES;")
databases = cur.fetchall()
print([db[0] for db in databases])

cur.close()
conn.close()


# ------------------------------------------------------------------------------------------------------------------------------
# [DataBase 생성/삭제]
conn = pymysql.connect(**connect_info)
cur = conn.cursor()

# DB 생성
db_name = 'test_db'
cur.execute(f"CREATE DATABASE IF NOT EXISTS {db_name};")

# DB 삭제
# cur.execute(f"DROP DATABASE {db_name};")

conn.commit()
cur.close()
conn.close()


############################################################################################################
# [DataBase 접근 및 Table 목록조희]
conn = pymysql.connect(**connect_info, database='test_db')
cur = conn.cursor()

cur.execute("SHOW TABLES;")
tables = cur.fetchall()
print([tb[0] for tb in tables])

cur.close()
conn.close()



# -------------------------------------------------------------------------------------------
# (Table 정의서)

# 특정 table만 조회
table_id = 'TB_M2N_C03_C_WMT030'

# (1)
sql = f"SHOW CREATE TABLE POSM2N.{table_id};"

# (2)
sql = f"DESCRIBE POSM2N.{table_id};"
columns = ['Field', 'Type', 'Null', 'Key', 'Default', 'Extra']

# (3)
sql = f"SHOW COLUMNS FROM POSM2N.{table_id};"
columns = ['Field', 'Type', 'Null', 'Key', 'Default', 'Extra']

# (4)
sql = f"""
SELECT TABLE_NAME, COLUMN_NAME, COLUMN_COMMENT, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY, COLUMN_DEFAULT, EXTRA
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = 'POSM2N'
  AND TABLE_NAME = '{table_id}'
ORDER BY ORDINAL_POSITION;
"""
columns = ['TABLE_NAME', 'COLUMN_NAME', 'COLUMN_COMMENT', 'COLUMN_TYPE', 'IS_NULLABLE', 'COLUMN_KEY', 'COLUMN_DEFAULT' , 'EXTRA']


# 전체 table - column 조회
sql = """
SELECT TABLE_NAME, COLUMN_NAME, COLUMN_COMMENT, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY, COLUMN_DEFAULT, EXTRA
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = 'POSM2N'
ORDER BY TABLE_NAME, ORDINAL_POSITION;
"""
columns = ['TABLE_NAME', 'COLUMN_NAME', 'COLUMN_COMMENT', 'COLUMN_TYPE', 'IS_NULLABLE', 'COLUMN_KEY', 'COLUMN_DEFAULT' , 'EXTRA']


# Execution
conn = pymysql.connect(**connect_info, database='POSM2N')
cur = conn.cursor()
cur.execute(sql)
table_info = cur.fetchall()

result_table_info = pd.DataFrame(table_info, columns=columns)
cur.close()
conn.close()

result_table_info

# -------------------------------------------------------------------------------------------






############################################################################################################
# [Data 조회]

# ------------------------------------------------------------------------------------------------------------
sql = """
SELECT TABLE_NAME, COLUMN_NAME, COLUMN_COMMENT, COLUMN_TYPE, IS_NULLABLE, COLUMN_KEY, COLUMN_DEFAULT, EXTRA
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = 'POSM2N'
ORDER BY TABLE_NAME, ORDINAL_POSITION;
"""

# ------------------------------------------------------------------------------------------------------------

file_path = 'D:/DataScience/PythonForwork/MariaDB/Query_PosFrame_1CAL_ELT1_TEN_DIFF.sql'
file = open(file_path, 'r',encoding='UTF8')
sql = file.read()
# ------------------------------------------------------------------------------------------------------------



conn = pymysql.connect( ** connect_info)
with conn.cursor() as cs:
    cs.execute(sql)
    descriptions = cs.description
    desc_cols = ('name', 'type_code', 'display_size', 'internal_size', 'precision', 'scale', 'null_ok')
    
    descriptioins_df = pd.DataFrame(descriptions, columns=desc_cols)
    columns = list(map(lambda x: x[0], descriptions))
    
    results = cs.fetchall()
    results_df = pd.DataFrame(results, columns=columns)

results_df
conn.close()



# (descriptions)
#   name - 컬럼 이름 (예: TB_M2N_KC42CAL_MIC_SHEAR_L_4MD_IOW20627_5)
#   type_code - 컬럼의 데이터 타입 코드 (DB 내부 타입을 나타내는 숫자)
#   display_size - 컬럼을 표시할 때의 최대 크기 (일반적으로 None)
#   internal_size - 컬럼의 내부 저장 크기 (바이트 단위)
#   precision - 숫자형 컬럼의 정밀도 (소수점 포함 자리수)
#   scale - 숫자형 컬럼의 소수점 이하 자리수
#   null_ok - NULL 값을 허용하는지 여부 (True/False)

# ( type_code )
# {
# 'DECIMAL'      : 0
# ,'TINY'         : 1
# ,'SHORT'        : 2
# ,'LONG'         : 3
# ,'FLOAT'        : 4
# ,'DOUBLE'       : 5
# ,'NULL'         : 6
# ,'TIMESTAMP'    : 7
# ,'LONGLONG'     : 8
# ,'INT24'        : 9
# ,'DATE'         : 10
# ,'TIME'         : 11
# ,'DATETIME'     : 12
# ,'YEAR'         : 13
# ,'NEWDATE'      : 14
# ,'VARCHAR'      : 15
# ,'BIT'          : 16
# ,'JSON'         : 245
# ,'NEWDECIMAL'   : 246
# ,'ENUM'         : 247
# ,'SET'          : 248
# ,'TINY_BLOB'    : 249
# ,'MEDIUM_BLOB'  : 250
# ,'LONG_BLOB'    : 251
# ,'BLOB'         : 252
# ,'VAR_STRING'   : 253
# ,'STRING'       : 254
# ,'GEOMETRY'     : 255
# }


############################################################################################################
# 테이블 생성 / 삭제 / 수정 / 조회
def tb_list(cur):
    cur.execute("SHOW TABLES;")
    tables = cur.fetchall()
    return [tb[0] for tb in tables]

conn = pymysql.connect(**connect_info, database='test_db')
cur = conn.cursor()

# 테이블 생성
cur.execute("""
CREATE TABLE IF NOT EXISTS test_table (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    department VARCHAR(50),
    hire_date DATE
);
""")
print(tb_list(cur))

# 데이터 삽입
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
columns = [desc[0] for desc in cur.description]
print(pd.DataFrame(rows, columns=columns))

# 테이블 삭제
cur.execute("DROP TABLE test_table;")
print(tb_list(cur))

conn.commit()
cur.close()
conn.close()


# ------------------------------------------------------------------------------------------------------------------------------
# [TABLE KEY]
#     PRIMARY KEY	 : 테이블당 1개	(행 고유 식별)	무결성보장 O	자동인덱스생성 O
#     FOREIGN KEY	 : 여러 개 가능	(다른 테이블 참조)	무결성보장 O	자동인덱스생성 X
#     INDEX(Secondary Key)	: 여러 개 가능	(검색 속도 향상)	무결성보장 X	자동인덱스생성 O(검색용)

conn = pymysql.connect(**connect_info, database='test_db')
cur = conn.cursor()

sql_script = """
CREATE TABLE IF NOT EXISTS department (
    dept_id INT AUTO_INCREMENT PRIMARY KEY,
    dept_name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS employee (
    emp_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department_id INT NOT NULL,
    hire_date DATE NOT NULL,
    salary DECIMAL(10, 2),
    FOREIGN KEY (department_id) REFERENCES department(dept_id)
);

CREATE INDEX idx_employee_name ON employee(name);
CREATE INDEX idx_employee_hire_date ON employee(hire_date);
"""

for statement in sql_script.strip().split(';'):
    if statement.strip():
        cur.execute(statement)

conn.commit()
print(tb_list(cur))

# 샘플 데이터 삽입
departments = [("철강사업부",), ("에너지사업부",), ("연구개발부",)]
cur.executemany("INSERT IGNORE INTO department (dept_name) VALUES (%s);", departments)

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

# JOIN 조회
join_script = """
SELECT e.emp_id, e.name, d.dept_name, e.hire_date, e.salary
FROM employee e
JOIN department d ON e.department_id = d.dept_id
ORDER BY e.emp_id;
"""
cur.execute(join_script)
rows = cur.fetchall()
columns = [desc[0] for desc in cur.description]
print(pd.DataFrame(rows, columns=columns))

# 테이블 삭제
cur.execute("DROP TABLE employee;")
cur.execute("DROP TABLE department;")
conn.commit()

cur.close()
conn.close()