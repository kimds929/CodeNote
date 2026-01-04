import psycopg2
from psycopg2 import sql
import pandas as pd

### TBD (23.11.16.)
# 1. Activities DB에서 row (혹은 Information) 뽑아내는 함수.
# 2. User DB에서 Feature 뽑아내는 함수. (1과 동일한 함수 사용할 수도 있고 따로 구현할 수도 있고.)
# 3. Feedback learning으로 얻은 정보를 User DB에 추가하는 함수.

# Server connection info

# Schema (TBD)

# Initialize connection

class pg_connection():
    def __init__(self):
        # Pre-Connection Info
        self.__HOST = '127.0.0.1'
        self.__DBNAME = 'test_db'
        self.__USER = 'postgres'
        self.__PASSWORD = '****'
        self.__PORT = '5432'
        
        # Connection Variables
        self.conn = None
        self.cursor = None
        
        # Initialize connection
        self.initialize_connection()
        
    def initialize_connection(self):    
        print(f"\nCreate connection to Postgres...")
        try:
            connection_info = "host={} dbname={} user={} password={} port={}".format(
                self.__HOST, self.__DBNAME, self.__USER, self.__PASSWORD, self.__PORT)
            self.conn = psycopg2.connect(connection_info)
            self.cursor = self.conn.cursor()
            print("\nSuccess in connecting to Postgres!")
        except psycopg2.Error as e:
            print("\nPostgres Error: ", e)
            
    def disconnect(self):
        if self.conn != None:
            self.cursor.close()
            self.conn.close()
        print("Successfully Disconnected Postgres")

    # Create or replace table (TBD: Is it really needed?)
    def create_table(self, table_name):
        return "something"

    # Fetch table of the table name as Pandas dataframe
    def fetch_table(self, table_name, columns="*", where=None):
        sql_query = "SELECT {} FROM \"{}\";".format(columns, table_name)        
        if where:
            sql_query = "SELECT {} FROM \"{}\" WHERE {};".format(columns, table_name, where)
        df = pd.read_sql(sql_query, self.conn)
        return df
    
    def index_search(self, table_name, ids):
        sql_query = "SELECT * from {} where \"id\" in {}".format(table_name, tuple(ids))
        self.cursor.execute(sql_query)
        result = self.cursor.fetchall()
        return result

