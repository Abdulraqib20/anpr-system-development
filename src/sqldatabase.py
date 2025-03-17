import psycopg2
from psycopg2.pool import SimpleConnectionPool
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.resolve()))  # More reliable path resolution

from config.appconfig import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, DB_PORT

# Database connection parameters
db_params = {
    "host": DB_HOST,
    "database": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "port": DB_PORT
}

connection_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    **db_params
)

def get_connection():
    return connection_pool.getconn()

def release_connection(conn):
    connection_pool.putconn(conn)

def create_connection(db_params):
    """Create a database connection to the PostgreSQL database"""
    conn = None
    try:
        conn = psycopg2.connect(**db_params)
        print(f"Connected to PostgreSQL database successfully.")
    except (Exception, psycopg2.Error) as error:
        print(f"Error connecting to PostgreSQL database: {error}")
    return conn

# def create_table(conn):
#     """Create the license_plates2 table if it doesn't exist"""
#     try:
#         with conn.cursor() as cursor:
#             cursor.execute('''
#                 CREATE TABLE IF NOT EXISTS license_plates2(
#                     id SERIAL PRIMARY KEY,
#                     start_time TIMESTAMP NOT NULL,
#                     end_time TIMESTAMP NOT NULL,
#                     license_plate TEXT NOT NULL,
#                     confidence FLOAT,
#                     detection_count INTEGER,
#                     vehicle_type TEXT,
#                     vehicle_color TEXT,
#                     time_of_day TEXT,
#                     day_of_week TEXT,
#                     UNIQUE(start_time, end_time, license_plate)
#                 )
#             ''')
#         conn.commit()
#         print("Table created successfully.")
#     except (Exception, psycopg2.Error) as error:
#         print(f"Error creating table: {error}")


def create_table_new(conn):
        """Create the detected_plates table if it doesn't exist"""
        try:
            with conn.cursor() as cursor:
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS detected_plates(
                        id SERIAL PRIMARY KEY,
                        start_time TIMESTAMP NOT NULL,
                        end_time TIMESTAMP NOT NULL,
                        license_plate VARCHAR(20) NOT NULL,
                        confidence FLOAT,
                        detection_count INTEGER,
                        vehicle_type TEXT,
                        vehicle_color TEXT,
                        time_of_day TEXT,
                        day_of_week TEXT,
                        UNIQUE(start_time, end_time, license_plate)
                    )
                ''')
            conn.commit()
            print("detected_plates table created successfully.")
        except (Exception, psycopg2.Error) as error:
            print(f"Error creating table: {error}")
       

   
def main():
    # Connect to the PostgreSQL database
    conn = create_connection(db_params)

    # If connection is successful, create the table
    if conn:
        create_table_new(conn)
        conn.close()
        print("Database connection closed.")

if __name__ == '__main__':
    main()