import sqlite3

MAX_CONVERSATIONS = 1000

def crear_base_datos():
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute('''
              CREATE TABLE IF NOT EXISTS conversations (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_input TEXT NOT NULL,
              chatbot_response TEXT NOT NULL)
              ''')
    conn.commit()
    conn.close()

def guardar_conversacion(user_input, chatbot_response):
    conn = sqlite3.connect('chatbot.db')
    c = conn.cursor()
    c.execute('''
              INSERT INTO conversations (user_input, chatbot_response)
              VALUES (?, ?)
              ''', (user_input, chatbot_response))
    conn.commit()
    c.execute('DELETE FROM conversations WHERE id NOT IN (SELECT id FROM conversations ORDER BY id DESC LIMIT ?)', (MAX_CONVERSATIONS,))
    conn.commit()
    conn.close()

