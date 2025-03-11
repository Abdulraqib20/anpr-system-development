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
    f.write('''CREATE TABLE LicensePlates(
        id SERIAL PRIMARY KEY,
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP NOT NULL,
        license_plate TEXT NOT NULL,
        confidence FLOAT,
        UNIQUE(start_time, end_time, license_plate)
    );\n\n''')

    # Write the INSERT statements for each row
    for row in rows:
        f.write(f"INSERT INTO LicensePlates (start_time, end_time, license_plate) VALUES ('{row[1]}', '{row[2]}', '{row[3]}');\n")

# Close the SQLite connection
sqlite_conn.close()

print("Export complete. Check licensePlates.sql for the output.")

#      psql -U postgres -d license -f db/licensePlates.sql
