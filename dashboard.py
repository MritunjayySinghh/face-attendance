import pandas as pd
import streamlit as st

import database

st.set_page_config(page_title="Lecture attendance", page_icon="📋", layout="wide")
st.title("📋 Lecture attendance")

db = database.connect()
lectures = pd.read_sql("SELECT * FROM lectures ORDER BY id DESC", db)
if lectures.empty:
    st.info("No lectures yet. Run attendance.py first!")
    st.stop()

# 1. Pick a lecture (newest first)
labels = {row.id: f"#{row.id} · {row.module} · {row.started}" for row in lectures.itertuples()}
lecture_id = st.selectbox("Lecture", list(labels), format_func=labels.get)
module = lectures.set_index("id").loc[lecture_id, "module"]
records = pd.read_sql("SELECT * FROM attendance WHERE lecture_id = ?", db, params=(lecture_id,))
records = records.fillna("-")
on_list = records[records.on_list == 1]

GROUPS = {
    "Present": "✅ Whole lecture",
    "Left early": "🚪 Left in between",
    "Partial": "⏰ Partly there",
    "Absent": "❌ Didn't show up",
}

# 2. The numbers at a glance
columns = st.columns(5)
columns[0].metric("👥 Supposed to be there", len(on_list))
for column, (status, title) in zip(columns[1:], GROUPS.items()):
    column.metric(title, int((on_list.status == status).sum()))

# 3. Who is in each group
for status, title in GROUPS.items():
    group = on_list[on_list.status == status]
    st.subheader(f"{title} ({len(group)})")
    if group.empty:
        st.caption("Nobody")
    else:
        st.dataframe(group[["name", "percent", "arrived", "last_seen"]], hide_index=True)

extras = records[records.on_list == 0]
if not extras.empty:
    st.subheader(f"➕ Not on the class list ({len(extras)})")
    st.dataframe(extras[["name", "status", "percent", "arrived", "last_seen"]], hide_index=True)

# 4. Everyone's record across all lectures of this module
st.divider()
st.subheader(f"📈 {module}: all lectures so far")
summary = pd.read_sql("""
    SELECT name,
           COUNT(*)                   AS "lectures",
           SUM(status = 'Present')    AS "whole lecture",
           SUM(status = 'Left early') AS "left early",
           SUM(status = 'Partial')    AS "partly there",
           SUM(status = 'Absent')     AS "missed"
    FROM attendance
    JOIN lectures ON lectures.id = attendance.lecture_id
    WHERE lectures.module = ? AND attendance.on_list = 1
    GROUP BY name
    ORDER BY name""", db, params=(module,))
summary["attendance %"] = (100 * summary["whole lecture"] / summary["lectures"]).round()
st.dataframe(summary, hide_index=True)