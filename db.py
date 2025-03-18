import sqlite3

# Open database connection
conn = sqlite3.connect("data.db")
print("Opened database successfully")

# Create users table if it doesn't exist
conn.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        email TEXT NOT NULL UNIQUE,
                        mobile TEXT NOT NULL,
                        password TEXT NOT NULL)''')  

# Create admin table if it doesn't exist
conn.execute('''CREATE TABLE IF NOT EXISTS admin (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        email TEXT NOT NULL UNIQUE,
                        password TEXT NOT NULL)''')  


# Create admin table if it doesn't exist
conn.execute('''CREATE TABLE IF NOT EXISTS faq (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subject TEXT NOT NULL,
                        answer TEXT)''')  

# Check if admin record exists
cursor = conn.execute('SELECT COUNT(*) FROM admin')
if cursor.fetchone()[0] == 0:
    # Insert an admin record if it doesn't exist
    conn.execute('''INSERT INTO admin (username, email, password) 
                    VALUES ('admin', 'admin@gmail.com', 'admin')''')
    conn.commit()  # Commit the transaction after insert

print("Tables created successfully")
conn.close()
