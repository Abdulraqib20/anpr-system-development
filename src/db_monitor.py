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
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def fetch_latest_records(conn):
    """Fetch the latest 100 records from the database."""
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT id, start_time, end_time, license_plate, confidence, vehicle_type, vehicle_color, time_of_day, day_of_week
            FROM license_plates2
            ORDER BY id ASC
        """)
        return cursor.fetchall()

def display_records(records):
    """Display records in a formatted, colorful table."""
    clear_screen()
    # ANSI escape codes for colors
    HEADER = "\033[95m"
    OKGREEN = "\033[92m"
    ENDC = "\033[0m"
    
    title = f"{HEADER}=== Real-Time License Plate Database Monitor ==={ENDC}"
    print(title)
    
    # Create a header line with colors
    header_line = (
        f"{HEADER}{'ID':<20} | {'Start Time':<20} | {'End Time':<20} | {'Number Plate':<15} | "
        f"{'Confidence':<10} | {'Vehicle Type':<15} | {'Vehicle Color':<15} | "
        f"{'Time of Day':<12} | {'Day of Week':<12}{ENDC}"
    )
    print(header_line)
    print("-" * 130)
    
    # Display each record in a neat row
    for record in records:
        id, start_time, end_time, license_plate, confidence, vehicle_type, vehicle_color, time_of_day, day_of_week = record
        row = (
            f"{str(id):<20} | {str(start_time):<20} | {str(end_time):<20} | {license_plate:<15} | "
            f"{confidence:<10.2f} | {vehicle_type:<15} | {vehicle_color:<15} | "
            f"{time_of_day:<12} | {day_of_week:<12}"
        )
        print(row)

def main():
    """Main monitoring loop."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        print("Connected to database. Monitoring updates... (Press Ctrl+C to exit)")
        
        while True:
            records = fetch_latest_records(conn)
            display_records(records)
            time.sleep(10)  # Refresh every 10 seconds
            
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    main()