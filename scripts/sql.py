import sqlite3

# 1. CONNECT (creates the file if it doesn't exist)
conn = sqlite3.connect("example.db")

# 2. CREATE A TABLE
conn.execute(
    """
CREATE TABLE IF NOT EXISTS parameters (
    id INTEGER PRIMARY KEY,
    name TEXT,
    value INTEGER
)
"""
)

# 3. INSERT SOME DATA
conn.execute("INSERT INTO people (name, age) VALUES (?, ?)", ("Alice", 30))
conn.execute("INSERT INTO people (name, age) VALUES (?, ?)", ("Bob", 24))

conn.commit()  # save changes

# 4. READ THE DATA
cursor = conn.execute("SELECT id, name, age FROM people")
for row in cursor:
    print(row)

# 5. CLOSE
conn.close()
