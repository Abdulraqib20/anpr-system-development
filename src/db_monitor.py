import psycopg2
import time
from dotenv import load_dotenv
from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(str(Path(__file__).parent.parent.resolve()))
sys.path.append(os.path.abspath("src"))

from config.appconfig import (
    DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT,
)

DB_CONFIG = {
    "host": DB_HOST,
    "database": DB_NAME,
    "user": DB_USER,
    "password": DB_PASSWORD,
    "port": DB_PORT
}

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def fetch_latest_records(conn):
    """Fetch the latest 10 records from the database"""
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT id, start_time, end_time, license_plate, confidence, detection_count
            FROM license_plates
            ORDER BY end_time DESC
            LIMIT 10
        """)
        return cursor.fetchall()

def display_records(records):
    """Display records in a formatted table"""
    clear_screen()
    print("=== Real-Time License Plate Database Monitor ===")
    print(f"{'ID':<5} | {'Start Time':<20} | {'End Time':<20} | {'Plate':<10} | {'Confidence':<10} | {'Detection Count':<10}")
    print("-" * 70)
    for record in records:
        print(f"{record[0]:<5} | {str(record[1]):<20} | {str(record[2]):<20} | {record[3]:<10} | {record[4]:<10.2f} | {record[5]:<10}")

def main():
    """Main monitoring loop"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Connected to database. Monitoring updates... (Ctrl+C to exit)")
        
        while True:
            records = fetch_latest_records(conn)
            display_records(records)
            time.sleep(5)  # Refresh every 5 second
            
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    main()