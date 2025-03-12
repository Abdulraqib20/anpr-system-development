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
# sys.path.append(os.path.abspath("src"))

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

def create_table(conn):
    """Create the license_plates table if it doesn't exist"""
    try:
        with conn.cursor() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS license_plates(
                    id SERIAL PRIMARY KEY,
                    start_time TIMESTAMP NOT NULL,
                    end_time TIMESTAMP NOT NULL,
                    license_plate TEXT NOT NULL,
                    confidence FLOAT,
                    detection_count INTEGER,
                    UNIQUE(start_time, end_time, license_plate)
                )
            ''')
        conn.commit()
        print("Table created successfully.")
    except (Exception, psycopg2.Error) as error:
        print(f"Error creating table: {error}")
        
# def save_to_database(license_plates, start_time, end_time):
#     conn = create_connection(db_params)
#     if conn:
#         cursor = conn.cursor()
#         for plate in license_plates:
#             try:
#                 cursor.execute('''
#                     INSERT INTO license_plates 
#                     (start_time, end_time, license_plate, confidence)
#                     VALUES (%s, %s, %s, %s)
#                     ON CONFLICT (start_time, end_time, license_plate) 
#                     DO NOTHING 
#                 ''', (start_time.isoformat(), end_time.isoformat(), plate[0], plate[1]))  # If plate is a tuple
#             except psycopg2.Error as e:
#                 print(f"Error inserting data: {e}")
#         conn.commit()
#         print("Data saved to the database successfully.")
#         conn.close()

def main():
    # Connect to the PostgreSQL database
    conn = create_connection(db_params)

    # If connection is successful, create the table
    if conn:
        create_table(conn)
        conn.close()
        print("Database connection closed.")

if __name__ == '__main__':
    main()