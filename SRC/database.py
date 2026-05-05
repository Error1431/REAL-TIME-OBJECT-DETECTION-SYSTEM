import sqlite3
import datetime
import threading
import os

DB_PATH = 'stats.db'
lock = threading.Lock()

def init_db():
    with lock:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event_type TEXT,
                track_id INTEGER,
                class_name TEXT
            )
        ''')
        conn.commit()
        conn.close()

def log_event(event_type, track_id, class_name):
    # event_type can be "ENTRY", "EXIT", or "DETECT"
    now = datetime.datetime.now().isoformat()
    with lock:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO events (timestamp, event_type, track_id, class_name)
            VALUES (?, ?, ?, ?)
        ''', (now, event_type, track_id, class_name))
        conn.commit()
        conn.close()

def get_recent_history(minutes=60):
    # Returns counts of ENTRY and EXIT per minute for the last X minutes
    time_threshold = (datetime.datetime.now() - datetime.timedelta(minutes=minutes)).isoformat()
    
    with lock:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # We group by the minute string: "YYYY-MM-DDTHH:MM" (first 16 chars)
        cursor.execute('''
            SELECT substr(timestamp, 1, 16) as minute, event_type, COUNT(*) as count
            FROM events
            WHERE timestamp >= ? AND event_type IN ('ENTRY', 'EXIT')
            GROUP BY minute, event_type
            ORDER BY minute ASC
        ''', (time_threshold,))
        
        rows = cursor.fetchall()
        conn.close()
        
    # Format data for Chart.js
    labels = []
    entries = []
    exits = []
    
    # Simple mapping
    data_map = {}
    for r in rows:
        m = r['minute']
        if m not in data_map:
            data_map[m] = {"ENTRY": 0, "EXIT": 0}
        data_map[m][r['event_type']] = r['count']
        
    for m in sorted(data_map.keys()):
        labels.append(m[-5:]) # Just show HH:MM
        entries.append(data_map[m]["ENTRY"])
        exits.append(data_map[m]["EXIT"])
        
    return {
        "labels": labels,
        "entries": entries,
        "exits": exits
    }

# Initialize on load
init_db()
