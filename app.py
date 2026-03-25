from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse, Response
from starlette.middleware.sessions import SessionMiddleware
import threading
import psycopg2
import csv
import io
from datetime import datetime

# PDF
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

from camera_loop import start_camera_loop, stop_camera_loop, generate_frames

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key="supersecretkey")

thread = None

USERS = {
    "sreejith": "1234",
    "admin": "admin"
}

# ---------------- DB ----------------
def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="face_monitoring",
        user="postgres",
        password="2005"
    )

# ---------------- WORK DETAILS ----------------
def get_user_work_details(username):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT login_time, logout_time
        FROM attendance_logs
        WHERE name=%s
        ORDER BY login_time DESC
        LIMIT 1
    """, (username,))
    att = cursor.fetchone()

    cursor.execute("""
        SELECT leave_time, return_time, duration_seconds
        FROM break_logs
        WHERE name=%s
    """, (username,))
    breaks = cursor.fetchall()

    total_break = sum([b[2] for b in breaks if b[2]])

    total_work = 0
    if att:
        login, logout = att
        end = logout if logout else datetime.now()
        total_work = (end - login).seconds - total_break

    hours = total_work / 3600
    mcp = (hours / 8) * 100 if hours > 0 else 0

    return {
        "total_work_hours": round(hours, 2),
        "mcp": round(mcp, 2),
        "check_in": len([b for b in breaks if b[1]]),
        "check_out": len(breaks),
        "breaks": breaks
    }

# ---------------- LOGIN ----------------
@app.get("/")
def login_page(request: Request):
    if request.session.get("user"):
        return RedirectResponse("/dashboard", 303)
    return HTMLResponse(open("templates/login.html", encoding="utf-8").read())

@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username in USERS and USERS[username] == password:
        request.session["user"] = username
        return RedirectResponse("/dashboard", 303)
    return HTMLResponse("Invalid Login")

@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    res = RedirectResponse("/", 303)
    res.delete_cookie("session")
    return res

# ---------------- DASHBOARD ----------------
@app.get("/dashboard")
def dashboard(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/", 303)
    return HTMLResponse(open("templates/index.html", encoding="utf-8").read())

# ---------------- CAMERA ----------------
@app.get("/start")
def start(request: Request):
    global thread
    if thread is None or not thread.is_alive():
        thread = threading.Thread(target=start_camera_loop, daemon=True)
        thread.start()
    return {"status": "started"}

@app.get("/stop")
def stop():
    stop_camera_loop()
    return {"status": "stopped"}

@app.get("/video")
def video():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# ---------------- REPORT ----------------
@app.get("/reports")
def reports():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT name, leave_time, return_time, duration_seconds
        FROM break_logs
        WHERE DATE(leave_time)=CURRENT_DATE
        ORDER BY leave_time DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    table = "<table border=1><tr><th>Name</th><th>Leave</th><th>Return</th><th>Duration</th></tr>"
    for r in rows:
        table += f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td></tr>"
    table += "</table>"

    return HTMLResponse(f"""
    <h2>Reports</h2>
    <button onclick="window.location.href='/download'">CSV</button>
    <button onclick="window.location.href='/download-pdf'">PDF</button>
    {table}
    """)

# ---------------- CSV ----------------
@app.get("/download")
def download():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM break_logs")
    rows = cur.fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Name","Leave","Return","Duration"])
    writer.writerows(rows)

    return Response(output.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=report.csv"})

# ---------------- PDF ----------------
@app.get("/download-pdf")
def download_pdf():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM break_logs")
    rows = cur.fetchall()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("Face Monitoring Report", styles['Title']))

    data = [["Name","Leave","Return","Duration"]]
    for r in rows:
        data.append([str(x) for x in r])

    table = Table(data)
    table.setStyle(TableStyle([("GRID",(0,0),(-1,-1),1,colors.black)]))
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)

    return Response(buffer.getvalue(),
        media_type="application/pdf",
        headers={"Content-Disposition":"attachment; filename=report.pdf"})
@app.get("/user-details")
def user_details(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/", 303)

    username = request.session.get("user")
    data = get_user_work_details(username)

    return HTMLResponse(f"""
    <h2>👤 My Stats - {username}</h2>

    <p><b>Total Work Hours:</b> {data['total_work_hours']} hrs</p>
    <p><b>MCP:</b> {data['mcp']} %</p>
    <p><b>Break Count:</b> {data['check_out']}</p>
    <p><b>Returned Count:</b> {data['check_in']}</p>

    <h3>Break Details</h3>
    <table border=1>
    <tr><th>Leave</th><th>Return</th><th>Duration</th></tr>
    {"".join([
        f"<tr><td>{b[0]}</td><td>{b[1] or '-'}</td><td>{b[2] or 0}</td></tr>"
        for b in data['breaks']
    ])}
    </table>

    <br><a href="/dashboard">⬅ Back</a>
    """)
@app.get("/search-user")
def search_user_page():
    return HTMLResponse("""
    <h2>🔍 Search User</h2>
    <form method="post" action="/search-user">
        <input type="text" name="username" placeholder="Enter username" required>
        <button type="submit">Search</button>
    </form>
    """)
@app.post("/search-user")
def search_user(username: str = Form(...)):
    data = get_user_work_details(username)

    return HTMLResponse(f"""
    <h2>Results for {username}</h2>

    <p><b>Total Work Hours:</b> {data['total_work_hours']} hrs</p>
    <p><b>MCP:</b> {data['mcp']} %</p>

    <h3>Break Details</h3>
    <table border=1>
    <tr><th>Leave</th><th>Return</th><th>Duration</th></tr>
    {"".join([
        f"<tr><td>{b[0]}</td><td>{b[1] or '-'}</td><td>{b[2] or 0}</td></tr>"
        for b in data['breaks']
    ])}
    </table>

    <br><a href="/dashboard">⬅ Back</a>
    """)