import sqlite3
from langchain_community.utilities import SQLDatabase

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('example.db')

# Create a cursor object
cur = conn.cursor()

# Create a new table
cur.execute('''
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    department TEXT
)
''')

# Commit the changes
conn.commit()

# Insert data into the table
cur.execute("INSERT INTO employees (name, age, department) VALUES ('Alice', 30, 'HR')")
cur.execute("INSERT INTO employees (name, age, department) VALUES ('Bob', 24, 'Engineering')")
cur.execute("INSERT INTO employees (name, age, department) VALUES ('Charlie', 28, 'Marketing')")

# Commit the changes
conn.commit()

db = SQLDatabase.from_uri("sqlite:///example.db")

# Print the database dialect
print(db.dialect)

# Print usable table names
print(db.get_usable_table_names())