# POSTEGRESQL
import numpy as np
import psycopg2

np.array(dir(psycopg2))


# Set Information
host_ip = 'localhost'       # '127.0.0.1'
user = 'postegres'
password = "****"

# Connection Postgresql
conn = psycopg2.connect(host=host_ip, user=user, password=password)
conn.close()



# Create DB
db = psycopg2.connect(host='localhost', dbname='test_db', user='postgres', password='****', port=5432)

db.close()
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Connect to PostgreSQL DBMS

con = psycopg2.connect(user='postgres', password='****')
con.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

 # Obtain a DB Cursor
cursor          = con.cursor()
name_Database   = "test_db"

 # Create table statement
sqlCreateDatabase = "create database "+name_Database+";"


# Create a table in PostgreSQL database
cursor.execute(sqlCreateDatabase)
cursor.close()





# [PostegreSQL] ######################################################################################################

# https://www.bearpooh.com/146
# https://www.postgresql.org/
# https://www.postgresql.org/ftp/pgadmin/pgadmin4/v7.4/windows/     # (pgAdmin)

# pip install psycopg2 
# pip install psycopg2 --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org


# D:\PostegreSQL\DataBase

import psycopg2
# conn.close()
conn = psycopg2.connect(host="127.0.0.1", dbname="test_db",
                      user="postgres", password="****", port=5432)

cursor = conn.cursor()

# Create Table -----------------------------------------------------------
table_name = 'test_tb'
cursor.execute(f"DROP TABLE IF EXISTS {table_name}")  #Doping EMPLOYEE table if already exists.

#Creating table as per requirement
sql =f'''CREATE TABLE {table_name}(
   FIRST_NAME CHAR(20) NOT NULL,
   LAST_NAME CHAR(20),
   AGE INT,
   SEX CHAR(1),
   INCOME FLOAT
)'''

cursor.execute(sql)
print("Table created successfully........")
conn.commit()


# Table List ------------------------------------------------------------
cursor.execute("select relname from pg_class where relkind='r' and relname !~ '^(pg_|sql_)';")
cursor.fetchall()


# Delete Table -----------------------------------------------------------
table_name = 'employee'
cursor.execute(f"DROP TABLE {table_name}")
print("Table drop successfully........")
conn.commit()



# Exit ------------------------------------------------------------
cursor.close()
conn.close()







# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
import pandas as pd

tableName = 'test';
InputData = pd.read_clipboard();
InputData
exist_action = 'append'
InputData.to_sql(name=tableName, con=conn, if_exists=exist_action, index=False);  # 'test'라는 이름으로 InputData DataFrame 객체를 SQL에 저장
# if_exists='append' 옵션이 있으면, 기존 테이블에 데이터를 추가로 넣음
# if_exists='fail' 옵션이 있으면, 기존 테이블이 있을 경우, 아무일도 하지 않음
# if_exists='replace' 옵션이 있으면, 기존 테이블이 있을 경우, 기존 테이블을 삭제하고, 다시 테이블을 만들어서, 새로 데이터를 넣음







