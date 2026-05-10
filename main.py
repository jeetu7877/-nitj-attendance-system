"""
NITJ AI Attendance System - FastAPI Backend
Requirements:
    pip install fastapi uvicorn python-multipart face_recognition pillow numpy

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import io
import base64
import pickle
import sqlite3
import logging
from datetime import date, datetime
from typing import Optional

import numpy as np
from PIL import Image
import face_recognition

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Config ─────────────────────────────────────────────────────────────────────
DB_PATH   = "attendance.db"
PKL_PATH  = "faces.pkl"
LOG_LEVEL = logging.INFO

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="NITJ Attendance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Database ───────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur  = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            roll        TEXT    NOT NULL UNIQUE,
            branch      TEXT    NOT NULL,
            year        INTEGER NOT NULL,
            section     TEXT    NOT NULL,
            email       TEXT,
            face_enrolled INTEGER DEFAULT 0,
            created_at  TEXT    DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS attendance (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  INTEGER NOT NULL,
            roll        TEXT    NOT NULL,
            name        TEXT    NOT NULL,
            branch      TEXT    NOT NULL,
            year        INTEGER NOT NULL,
            section     TEXT    NOT NULL,
            date        TEXT    NOT NULL,
            time        TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            subject     TEXT    DEFAULT '',
            FOREIGN KEY (student_id) REFERENCES students(id),
            UNIQUE(roll, date)       -- one attendance per student per day
        );
    """)
    conn.commit()
    conn.close()
    log.info("Database initialised at %s", DB_PATH)

init_db()

# ── Face store helpers ─────────────────────────────────────────────────────────
def load_faces() -> dict:
    """Return {roll: encoding_array} from disk."""
    if os.path.exists(PKL_PATH):
        with open(PKL_PATH, "rb") as f:
            return pickle.load(f)
    return {}

def save_faces(store: dict):
    with open(PKL_PATH, "wb") as f:
        pickle.dump(store, f)

def decode_b64_image(b64: str) -> np.ndarray:
    """Decode a base64 JPEG/PNG string → RGB numpy array."""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)

# ── Pydantic models ────────────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    name:    str
    roll:    str
    branch:  str
    year:    int
    section: str
    email:   Optional[str] = ""
    image:   str           # base64 encoded frame from webcam

class RecognizeRequest(BaseModel):
    image:   str           # base64 encoded frame
    subject: Optional[str] = ""

# ── POST /register_student ─────────────────────────────────────────────────────
@app.post("/register_student")
async def register_student(req: RegisterRequest):
    # 1. Decode image
    try:
        rgb = decode_b64_image(req.image)
    except Exception as e:
        raise HTTPException(400, f"Invalid image data: {e}")

    # 2. Detect face
    locations = face_recognition.face_locations(rgb, model="hog")
    if not locations:
        raise HTTPException(400, "No face detected in the image. Please try again with better lighting.")
    if len(locations) > 1:
        raise HTTPException(400, "Multiple faces detected. Please ensure only one person is in frame.")

    # 3. Encode face
    encodings = face_recognition.face_encodings(rgb, locations)
    if not encodings:
        raise HTTPException(400, "Could not generate face encoding. Please try again.")
    encoding = encodings[0]

    # 4. Persist to SQLite
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO students (name, roll, branch, year, section, email, face_enrolled)
            VALUES (?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(roll) DO UPDATE SET
                name=excluded.name, branch=excluded.branch, year=excluded.year,
                section=excluded.section, email=excluded.email, face_enrolled=1
        """, (req.name, req.roll, req.branch, req.year, req.section, req.email))
        conn.commit()
    except Exception as e:
        conn.close()
        raise HTTPException(500, f"DB error: {e}")
    finally:
        conn.close()

    # 5. Save face encoding
    store = load_faces()
    store[req.roll] = {"encoding": encoding, "name": req.name, "branch": req.branch,
                       "year": req.year, "section": req.section}
    save_faces(store)

    log.info("Registered student %s (%s) with face encoding", req.name, req.roll)
    return {"success": True, "message": f"Student {req.name} registered successfully with face data."}

# ── POST /recognize ────────────────────────────────────────────────────────────
@app.post("/recognize")
async def recognize(req: RecognizeRequest):
    store = load_faces()
    if not store:
        raise HTTPException(400, "No students enrolled yet. Please register students first.")

    # 1. Decode image
    try:
        rgb = decode_b64_image(req.image)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    # 2. Detect faces
    locations = face_recognition.face_locations(rgb, model="hog")
    if not locations:
        return {"status": "no_face", "message": "No face detected in frame."}

    # 3. Encode faces
    encodings = face_recognition.face_encodings(rgb, locations)
    results = []

    known_rolls      = list(store.keys())
    known_encodings  = [store[r]["encoding"] for r in known_rolls]

    for enc in encodings:
        distances = face_recognition.face_distance(known_encodings, enc)
        best_idx  = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        confidence = round((1 - best_dist) * 100, 1)

        THRESHOLD = 0.52          # tune lower = stricter
        if best_dist > THRESHOLD:
            results.append({"status": "unknown", "confidence": confidence, "message": "Unknown person"})
            continue

        roll = known_rolls[best_idx]
        meta = store[roll]
        today = date.today().isoformat()
        now   = datetime.now().strftime("%H:%M:%S")

        # 4. Mark attendance (UNIQUE roll+date prevents duplicates)
        conn = get_db()
        try:
            # Fetch student id
            row = conn.execute("SELECT id FROM students WHERE roll=?", (roll,)).fetchone()
            if not row:
                conn.close()
                results.append({"status": "error", "message": "Student record missing"})
                continue
            student_id = row["id"]

            conn.execute("""
                INSERT INTO attendance (student_id, roll, name, branch, year, section, date, time, confidence, subject)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (student_id, roll, meta["name"], meta["branch"], meta["year"],
                  meta["section"], today, now, confidence, req.subject or ""))
            conn.commit()

            results.append({
                "status":     "present",
                "roll":       roll,
                "name":       meta["name"],
                "branch":     meta["branch"],
                "year":       meta["year"],
                "section":    meta["section"],
                "date":       today,
                "time":       now,
                "confidence": confidence,
            })
            log.info("Attendance marked: %s (%s) %.1f%%", meta["name"], roll, confidence)

        except sqlite3.IntegrityError:
            conn.close()
            results.append({
                "status":     "duplicate",
                "roll":       roll,
                "name":       meta["name"],
                "confidence": confidence,
                "message":    f"{meta['name']} already marked present today.",
            })
            continue
        except Exception as e:
            conn.close()
            results.append({"status": "error", "message": str(e)})
            continue
        finally:
            try:
                conn.close()
            except Exception:
                pass

    return {"results": results}

# ── GET /students ──────────────────────────────────────────────────────────────
@app.get("/students")
async def get_students(branch: Optional[str] = None, year: Optional[int] = None,
                       section: Optional[str] = None):
    conn = get_db()
    q  = "SELECT * FROM students WHERE 1=1"
    params = []
    if branch:  q += " AND branch=?";  params.append(branch)
    if year:    q += " AND year=?";    params.append(year)
    if section: q += " AND section=?"; params.append(section)
    q += " ORDER BY branch, year, section, name"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ── GET /attendance ────────────────────────────────────────────────────────────
@app.get("/attendance")
async def get_attendance(date_filter: Optional[str] = None,
                         branch: Optional[str] = None,
                         roll: Optional[str] = None):
    conn = get_db()
    q  = "SELECT * FROM attendance WHERE 1=1"
    params = []
    if date_filter: q += " AND date=?";   params.append(date_filter)
    if branch:      q += " AND branch=?"; params.append(branch)
    if roll:        q += " AND roll=?";   params.append(roll)
    q += " ORDER BY date DESC, time DESC"
    rows = conn.execute(q, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

# ── GET /stats ─────────────────────────────────────────────────────────────────
@app.get("/stats")
async def get_stats():
    conn  = get_db()
    today = date.today().isoformat()
    total_students  = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    face_enrolled   = conn.execute("SELECT COUNT(*) FROM students WHERE face_enrolled=1").fetchone()[0]
    today_present   = conn.execute("SELECT COUNT(*) FROM attendance WHERE date=?", (today,)).fetchone()[0]
    total_records   = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
    conn.close()
    return {
        "total_students":  total_students,
        "face_enrolled":   face_enrolled,
        "today_present":   today_present,
        "total_records":   total_records,
        "date":            today,
    }

# ── DELETE /students/{roll} ────────────────────────────────────────────────────
@app.delete("/students/{roll}")
async def delete_student(roll: str):
    conn = get_db()
    conn.execute("DELETE FROM students WHERE roll=?", (roll,))
    conn.commit()
    conn.close()
    store = load_faces()
    store.pop(roll, None)
    save_faces(store)
    return {"success": True, "message": f"Student {roll} deleted."}

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}
