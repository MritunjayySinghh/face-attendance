import csv
import os
import sqlite3

DB_PATH = "data/attendance.db"
CLASS_LISTS = "data/class_lists.csv"


def connect():
    """Open the database, and create the tables the first time."""
    os.makedirs("data", exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.execute("""
        CREATE TABLE IF NOT EXISTS lectures (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            module  TEXT NOT NULL,
            started TEXT NOT NULL,
            ended   TEXT NOT NULL
        )""")
    db.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            lecture_id INTEGER NOT NULL REFERENCES lectures(id),
            name       TEXT NOT NULL,
            on_list    INTEGER NOT NULL,   -- 1 = supposed to be there, 0 = not on the class list
            status     TEXT NOT NULL,      -- Present / Left early / Partial / Absent
            percent    INTEGER NOT NULL,
            arrived    TEXT,
            last_seen  TEXT
        )""")
    return db


def save_lecture(module, started, ended, rows):
    """Save one lecture and everyone's attendance. Returns the lecture's id."""
    db = connect()
    with db:   # saves (commits) everything at the end of this block
        cursor = db.execute("INSERT INTO lectures (module, started, ended) VALUES (?, ?, ?)",
                            (module, started, ended))
        lecture_id = cursor.lastrowid
        db.executemany("INSERT INTO attendance VALUES (?, ?, ?, ?, ?, ?, ?)",
                       [(lecture_id, *row) for row in rows])
    db.close()
    return lecture_id


def class_list(module):
    """Who is supposed to be at this module's lectures (from data/class_lists.csv)."""
    if not os.path.exists(CLASS_LISTS):
        return []
    with open(CLASS_LISTS, newline="", encoding="utf-8-sig") as file:   # utf-8-sig: also works if Excel saved it
        return [row["name"].strip() for row in csv.DictReader(file)
                if row["module"].strip().upper() == module]