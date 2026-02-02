
import pymysql
from pymysql.cursors import DictCursor
from pymysql import err

from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


folder_path = f'D:/DataScience/PythonForwork/MariaDB'       # YOUR FOLDER PATH
account_path = f'{folder_path}/SingleStoreDB_Account_Public.json'
# account_path = f'{folder_path}/SingleStoreDB_Account.json'
# (SingleStoreDB_Account.json)
#   {"username": "USER_NAME", "password": "PASSWORD", "last_update": "YYYYMMDD"}

# -------------------------------------------------------------------------------------------------------------------
# Password 변경 유효기간 계산을 위한 함수
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

# Password 변경일을 Account.json에 수정하기 위한 함수
def refresh_password(account):
    account['last_update'] = datetime.now().strftime("%Y%m%d")
    with open(f"{account_path}", "w", encoding='utf-8') as file:
        json.dump(account, file)
# -------------------------------------------------------------------------------------------------------------------

#----------------------------------------------------------------------
with open(f"{account_path}", "r") as file:
    account = json.load(file)
#----------------------------------------------------------------------

check_refresh_password(account)


connect_info = {
    'host': '172.28.63.181',            # SingleStore 호스트 주소
    'port': 3306,                       # 포트 
    'user': account['username'],        # 사용자명
    'password': account['password']     # 비밀번호
}
############################################################################################################








############################################################################################################

# SingleStore 연결 설정
"""SingleStore 데이터베이스에 연결"""
conn = pymysql.connect( ** connect_info)
print("SingleStore 연결 성공!")

cur = conn.cursor()

############################################################################################################
# [DataBase 목록조희]
cur.execute("SHOW DATABASES;")
databases = cur.fetchall()
print([db[0] for db in databases])      # POSM2K : 열연, POSM2N : 냉연

cur.close()
conn.close()

############################################################################################################
# [DataBase 접근 및 Table 목록조희]
conn = pymysql.connect(**connect_info, database='POSM2N')
cur = conn.cursor()

cur.execute("SHOW TABLES;")
tables = cur.fetchall()
print(np.array([tb[0] for tb in tables]))
pd.Series([tb[0] for tb in tables]).to_clipboard()

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
sql ="""
SELECT c.TABLE_NAME, t.TABLE_COMMENT, c.COLUMN_NAME, c.COLUMN_COMMENT, c.COLUMN_TYPE, c.IS_NULLABLE, c.COLUMN_KEY, c.COLUMN_DEFAULT, c.EXTRA
FROM information_schema.COLUMNS c
    JOIN information_schema.TABLES t
    ON c.TABLE_SCHEMA = t.TABLE_SCHEMA
    AND c.TABLE_NAME = t.TABLE_NAME
WHERE c.TABLE_SCHEMA = 'POSM2N'
ORDER BY c.TABLE_NAME, c.ORDINAL_POSITION;
"""
columns = ['TABLE_NAME', 'TABLE_COMMENT', 'COLUMN_NAME', 'COLUMN_COMMENT', 'COLUMN_TYPE', 'IS_NULLABLE', 'COLUMN_KEY', 'COLUMN_DEFAULT' , 'EXTRA']

# Execution
conn = pymysql.connect(**connect_info, database='POSM2N')
cur = conn.cursor()
cur.execute(sql)
table_info = cur.fetchall()

result_table_info = pd.DataFrame(table_info, columns=columns)
cur.close()
conn.close()

result_table_info[['TABLE_NAME', 'TABLE_COMMENT']].drop_duplicates().to_clipboard()
# -------------------------------------------------------------------------------------------






############################################################################################################
# [Data 조회]
def posframe_to_singlestore(sql):
    return sql.replace('BIZ_DATA.', 'POSM2N.').replace('VI_M27_', 'TB_M2N_')

# ------------------------------------------------------------------------------------------------------------

# posframe_sql = "SELECT 'a' FROM DUAL;"     # dummy table
# sql = posframe_to_singlestore(posframe_sql)

sql = "SELECT 'a' FROM DUAL;-"     # dummy table
# ------------------------------------------------------------------------------------------------------------

# file_path = 'D:/DataScience/PythonForwork/MariaDB/Query_PosFrame_1CAL_ELT1_TEN_DIFF.sql'
file_path = f'{folder_path}/test_query.sql'
file = open(file_path, 'r',encoding='UTF8')

# sql = file.read()
sql = posframe_to_singlestore(file.read())
# open(file_path, 'w',encoding='UTF8').write(sql)   # 수정사항 파일에 반영


# print(sql)
# ------------------------------------------------------------------------------------------------------------



conn = pymysql.connect( ** connect_info)
with conn.cursor() as cs:
    try:
        cs.execute(sql)
        descriptions = cs.description
        desc_cols = ('name', 'type_code', 'display_size', 'internal_size', 'precision', 'scale', 'null_ok')
        
        descriptioins_df = pd.DataFrame(descriptions, columns=desc_cols)
        columns = list(map(lambda x: x[0], descriptions))
        
        results = cs.fetchall()
        results_df = pd.DataFrame(results, columns=columns)
    except err.OperationalError as e:
        print("OperationalError 발생:", e)
    except err.ProgrammingError as e:
        print("ProgrammingError 발생:", e)
    except err.DatabaseError as e:
        print("DatabaseError 발생:", e)
    except Exception as e:
        print("기타 에러:", e)

results_df
columns
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