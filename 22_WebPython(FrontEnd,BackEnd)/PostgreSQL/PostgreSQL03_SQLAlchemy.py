
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

conn = psycopg2.connect(**connect_info, database='postgres')
cur = conn.cursor() # cursor
cur.close()
conn.close()

# db_list
def db_list(cur):
    cur.execute("SELECT datname FROM pg_database;")
    databases = cur.fetchall()
    return [db for db in databases]  # database list

# tb_list
def tb_list(cur):
    cur.execute("""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public'
    ORDER BY table_name;
    """)
    tables = cur.fetchall()
    return [tb for tb in tables]  # table list

# execute_script
def retrieval_all_data(cur, script):
    cur.execute(script)
    rows = cur.fetchall()
    columns = [desc.name for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)

############################################################################################################
# 1. SQLAlchemy 기본 구조
#   SQLAlchemy는 크게 두 가지 방식이 있습니다.
#     . Core 방식: SQL문을 직접 작성하는 스타일 (엔진 + 메타데이터 + 테이블 객체)
#     . ORM 방식: Python 클래스와 DB 테이블을 매핑하여 객체지향적으로 다루는 스타일
#   여기서는 ORM 방식을 중심으로 설명드리겠습니다.


############################################################################################################
# 2. SQLAlchemy에서 PostgreSQL 연결
from sqlalchemy import create_engine, inspect, Column, Integer, String, Date
from sqlalchemy.orm import declarative_base, sessionmaker
import pandas as pd

# PostgreSQL 접속 정보
user_name =  account['user_name']
password =  account['password']
host = "127.0.0.1"
port = 5432
database = "postgres"

# SQLAlchemy 엔진 생성 - create_engine() : DB 연결을 위한 엔진 객체 생성
#       postgresql+psycopg2 : PostgreSQL 드라이버 지정
# engine = create_engine(f"postgresql+psycopg2://{user_name}:{password}@{host}:{port}/{database}")
engine = create_engine(f"postgresql+psycopg2://{user_name}:{password}@{host}:{port}/{database}")



############################################################################################################
# DB List
from sqlalchemy import text
with engine.connect() as conn:
    result = conn.execute(text("SELECT datname FROM pg_database;"))
    db_list = [row[0] for row in result]
    print(db_list)


############################################################################################################
# ORM 매핑(Table생성)을 위한 Base 클래스 생성 - declarative_base() : ORM 클래스들이 상속받을 기본 클래스
Base = declarative_base()

# 세션 팩토리 생성 - sessionmaker() : DB와의 트랜잭션을 관리하는 세션 생성
SessionLocal = sessionmaker(bind=engine)


# 테이블 정의 (ORM 클래스 정의) → 이 시점에 Base.metadata에 테이블 정보 등록
class TestTable(Base):
    __tablename__ = "test_table"  # 실제 DB 테이블명
    id = Column(Integer, primary_key=True, autoincrement=True)  # SERIAL과 동일
    name = Column(String(100))  # VARCHAR(100)
    department = Column(String(50))  # VARCHAR(50)
    hire_date = Column(Date)  # DATE
# Column() : 각 필드를 정의
# primary_key=True : 기본키 설정
# autoincrement=True : 자동 증가
# String(n) : 가변 길이 문자열
# Date : 날짜 타입

# (SQLAlchemy)	    (PostgreSQL)입	    (설명)
# ----------------------------------------------------
# Integer	        INTEGER	            4바이트 정수
# SmallInteger	    SMALLINT	        2바이트 정수
# BigInteger	    BIGINT	            8바이트 정수
# Numeric           (precision, scale)	NUMERIC	고정 소수점
# Float	            REAL / DOUBLE PRECISION	부동소수점
# String(length)	VARCHAR(length)	    가변 길이 문자열
# Text	            TEXT	            길이 제한 없는 문자열
# Date	            DATE	            날짜
# Time	            TIME	            시각
# DateTime	        TIMESTAMP	        날짜+시각
# Boolean	        BOOLEAN	T           RUE/FALSE
# LargeBinary	    BYTEA	            이진 데이터
# JSON	            JSON	            JSON 데이터
# ARRAY	            ARRAY	            배열 타입
# Enum	            ENUM	            열거형
# UUID	            UUID	            범용 고유 식별자


# 테이블 생성 : create_all() : Base에 정의된 모든 ORM 클래스에 해당하는 테이블 생성 → 등록된 모든 테이블 생성
Base.metadata.create_all(engine)
inspect(engine).get_table_names()  # Table list 조회


# 테이블 삭제 : drop_all() : 모든 테이블 삭제
# Base.metadata.drop_all(engine)
# inspect(engine).get_table_names()  # Table list 조회

# 세션 생성
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# 데이터 입력
new_row = TestTable(name="홍길동", department="R&D", hire_date="2024-06-01")
session.add(new_row)        # session.add() : 객체를 세션에 추가
session.commit()        # session.commit() : 변경사항을 DB에 반영

# 데이터 수정
# session.query() : ORM 객체를 이용한 조회
# .filter() : WHERE 조건
# .first() : 첫 번째 결과 반환
# .delete() : 조건에 맞는 행 삭제

row_to_update = session.query(TestTable).filter(TestTable.name == "홍길동").first()     
row_to_update.department = "기술연구소"
session.commit()

# 데이터 조회
rows = session.query(TestTable).all()
df = pd.DataFrame([r.__dict__ for r in rows])
df = df.drop(columns=["_sa_instance_state"])
print(df)


# 데이터 삭제
session.query(TestTable).filter(TestTable.name == "홍길동").delete()        
session.commit()

# 세션 종료
session.close()


############################################################################################################
# Query로 데이터 조회
from sqlalchemy import inspect, create_engine, MetaData, Table, select, text
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker

engine = create_engine(f"postgresql+psycopg2://{user_name}:{password}@{host}:{port}/{database}")

# Automap Base 생성 및 Reflection
Base = automap_base()   # # Automap Base 생성
Base.prepare(engine, reflect=True)  # # DB의 모든 테이블 구조를 읽어오기

# table list확인
inspect(engine).get_table_names() 

# 메타데이터 
metadata = MetaData()
Table("test_table", metadata, autoload_with=engine)


# ----------------------------------------------------------------------------------------------------------------------------------------
script = """SELECT * FROM test_table"""

# SQL script 조회 1
with engine.connect() as conn:
    cur = conn.execute(text(script))
    result_df = pd.DataFrame(cur.fetchall(), columns=cur.keys())

# SQL script 조회 2
session = sessionmaker(bind=engine)()
cur = session.execute(text(script))
result_df = pd.DataFrame(cur.fetchall(), columns=cur.keys())
session.close()

# SQL script 조회 3
result_df = pd.DataFrame([r.__dict__ for r in session.query(TestTable).all()]).drop('_sa_instance_state', axis=1)


# ----------------------------------------------------------------------------------------------------------------------------------------
def query_to_df(engine, sql, params=None):
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df

query_to_df(engine, script)
# ----------------------------------------------------------------------------------------------------------------------------------------






############################################################################################################
# Table가져와서 데이터 추가
from sqlalchemy import inspect, create_engine, MetaData, Table, select, text
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker

engine = create_engine(f"postgresql+psycopg2://{user_name}:{password}@{host}:{port}/{database}")

# table list확인
inspect(engine).get_table_names() 

# Automap Base 생성 및 Reflection
Base = automap_base()
Base.prepare(engine, reflect=True)

# 매핑된 클래스 가져오기
TestTable = Base.classes.test_table  # Base.classes['test_table']  : DB의 test_table 테이블과 매핑된 ORM 클래스

# Session 생성
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# 추가할 데이터 리스트
data_list = [
    TestTable(name="대조영", department="기술연구소", hire_date=datetime.date(2024, 6, 1)),
    TestTable(name="이순신", department="생산기술", hire_date=datetime.date(2024, 6, 2)),
    TestTable(name="강감찬", department="품질관리", hire_date=datetime.date(2024, 6, 3)),
    TestTable(name="유관순", department="R&D", hire_date=datetime.date(2024, 6, 4)),
    TestTable(name="장보고", department="해양사업", hire_date=datetime.date(2024, 6, 5)),
]

# 한 번에 추가
session.add_all(data_list)

# 커밋
session.commit()

# 세션 종료
session.close()


# 테이블 조회
# script = """SELECT * FROM test_table"""
script = """SELECT * FROM test_table where name='홍길동' """
query_to_df(engine, script)




############################################################################################################
# 데이터 수정 1 -------------------------------------------------------------------------------------------
session.query(TestTable).filter(TestTable.id == 2) \
    .update({TestTable.name: "대조영", TestTable.hire_date: datetime.date(2024, 6, 10)}, synchronize_session=False)
session.commit()

# 테이블 조회
script = """SELECT * FROM test_table"""
query_to_df(engine, script)

# 데이터 수정 2 -------------------------------------------------------------------------------------------
rows = session.query(TestTable).filter(TestTable.department.in_(["기술연구소", "R&D"])).all()
for row in rows:
    if row.department == "기술연구소":
        row.department = "신기술연구소"
    elif row.department == "R&D":
        row.department = "연구개발본부"
session.commit()

# 테이블 조회
script = """SELECT * FROM test_table"""
query_to_df(engine, script)



# 데이터 수정 3 : SQL문으로 조회해서 데이터 수정 -------------------------------------------------------------------------------------------
sql = text("""
    SELECT * FROM test_table
    WHERE department IN ('신기술연구소', '연구개발본부')
""")

# 실행 후 ORM 객체로 변환
cur = session.execute(sql)
rows = cur.fetchall()
print( pd.DataFrame(rows, columns=cur.keys()) )


# 수정은 ORM 객체로 다시 조회 후 진행
orm_rows = session.query(TestTable).from_statement(sql).all()
for row in orm_rows:
    if row.department == "신기술연구소":
        row.department = "기술연구소"
    elif row.department == "연구개발본부":
        row.department = "R&D"

session.commit()

script = """SELECT * FROM test_table"""
query_to_df(engine, script)























# ############################################################################################################
# # SQLAlchemy로 제어하기 (ORM 방식)
# from sqlalchemy import create_engine, Column, Integer, String, Date
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker

# # DB 연결
# engine = create_engine("postgresql://postgres:비밀번호@127.0.0.1:5432/my_new_db")
# Base = declarative_base()

# # 모델 정의
# class Employee(Base):
#     __tablename__ = 'employees'
#     id = Column(Integer, primary_key=True)
#     name = Column(String(100))
#     department = Column(String(50))
#     hire_date = Column(Date)

# # 테이블 생성
# Base.metadata.create_all(engine)

# # 세션 생성
# Session = sessionmaker(bind=engine)
# session = Session()

# # 데이터 입력
# emp = Employee(name="홍길동", department="R&D", hire_date="2024-06-01")
# session.add(emp)
# session.commit()

# # 데이터 조회
# for e in session.query(Employee).all():
#     print(e.id, e.name, e.department)

# # 데이터 수정
# emp.department = "기술연구소"
# session.commit()

# # 데이터 삭제
# session.delete(emp)
# session.commit()



