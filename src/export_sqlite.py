import sqlite3

# Connect to your SQLite database
sqlite_conn = sqlite3.connect('db/licensePlatesDatabase.db')
cursor = sqlite_conn.cursor()

# Fetch all data from LicensePlates table
cursor.execute("SELECT * FROM LicensePlates")
rows = cursor.fetchall()

# Open a file to write the SQL dump
with open('./db/licensePlates.sql', 'w') as f:
    # Write the CREATE TABLE statement
    f.write("""CREATE TABLE IF NOT EXISTS license_plates2 (
        id SERIAL PRIMARY KEY,
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP NOT NULL,
        license_plate VARCHAR(15) NOT NULL UNIQUE,
        confidence FLOAT,
        detection_count INTEGER,
        vehicle_type VARCHAR(20),
        vehicle_color VARCHAR(20),
        time_of_day VARCHAR(20),
        day_of_week VARCHAR(20)
    );\n\n""")
    
    # Batch insert with parameter substitution
    f.write("BEGIN TRANSACTION;\n")
    for row in rows:
        # Handle NULL values and proper quoting
        values = [
            f"'{value}'" if isinstance(value, str) else 
            str(value) if value is not None else 
            'NULL' 
            for value in row
        ]
        insert_sql = f"""INSERT INTO license_plates2 (
            start_time, end_time, license_plate, confidence,
            detection_count, vehicle_type, vehicle_color,
            time_of_day, day_of_week
        ) VALUES ({', '.join(values)});\n"""
        
        f.write(insert_sql)
    f.write("COMMIT;\n")

# Close the SQLite connection
sqlite_conn.close()

print("Export complete. Check licensePlates.sql for the output.")

#      psql -U postgres -d license -f db/licensePlates.sql
