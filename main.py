"""
NITJ Classroom Backend — FastAPI + SQLite
==========================================
Endpoints match the frontend exactly.
Face recognition: uses face_recognition lib if installed, else OpenCV HOG fallback.

Run:
    pip install fastapi uvicorn python-jose[cryptography] passlib[bcrypt] python-multipart aiofiles pillow numpy face_recognition
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os, json, base64, random, string, uuid, time, math, logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List
import sqlite3, shutil

from fastapi import (
    FastAPI, Depends, HTTPException, status,
    UploadFile, File, Form, Body, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, EmailStr
from passlib.context import CryptContext
from jose import JWTError, jwt
import numpy as np

# ── Try to import face_recognition; fall back to OpenCV ──────────────────────
try:
    import face_recognition as fr
    FACE_LIB = "face_recognition"
except ImportError:
    fr = None
    FACE_LIB = "opencv"

try:
    import cv2
    CV2_OK = True
except ImportError:
    cv2 = None
    CV2_OK = False

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nitj")

# ─────────────────────────── CONFIG ──────────────────────────────────────────
SECRET_KEY  = os.getenv("SECRET_KEY", "nitj-super-secret-key-change-in-prod-2025")
ALGORITHM   = "HS256"
TOKEN_EXPIRE_HOURS = 72
DB_PATH     = "nitj.db"
UPLOAD_DIR  = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

try:
    from passlib.handlers.bcrypt import bcrypt as _bc_test
    _bc_test.using(rounds=4).hash("test")
    pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
except Exception:
    pwd_ctx = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
app = FastAPI(title="NITJ Classroom API", version="9.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Serve frontend HTML
FRONTEND_HTML = Path("index.html")

# ─────────────────────────── DATABASE ────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def init_db():
    db = get_db()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        email       TEXT UNIQUE NOT NULL,
        password    TEXT NOT NULL,
        department  TEXT DEFAULT '',
        created_at  TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS classrooms (
        id           TEXT PRIMARY KEY,
        name         TEXT NOT NULL,
        subject      TEXT NOT NULL,
        branch       TEXT NOT NULL,
        year         INTEGER NOT NULL,
        section      TEXT NOT NULL,
        description  TEXT DEFAULT '',
        banner_color TEXT DEFAULT '#1255a6',
        code         TEXT UNIQUE NOT NULL,
        admin_id     TEXT NOT NULL,
        created_at   TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(admin_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS enrollments (
        id           TEXT PRIMARY KEY,
        classroom_id TEXT NOT NULL,
        user_id      TEXT NOT NULL,
        roll_number  TEXT DEFAULT '',
        branch       TEXT DEFAULT '',
        year         INTEGER DEFAULT 1,
        section      TEXT DEFAULT '',
        face_data    TEXT DEFAULT '',
        face_enrolled INTEGER DEFAULT 0,
        joined_at    TEXT DEFAULT (datetime('now')),
        UNIQUE(classroom_id, user_id),
        FOREIGN KEY(classroom_id) REFERENCES classrooms(id),
        FOREIGN KEY(user_id)      REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS attendance (
        id           TEXT PRIMARY KEY,
        classroom_id TEXT NOT NULL,
        student_id   TEXT NOT NULL,
        date         TEXT NOT NULL,
        time         TEXT NOT NULL,
        status       TEXT NOT NULL,
        confidence   REAL DEFAULT 0,
        created_at   TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(classroom_id) REFERENCES classrooms(id),
        FOREIGN KEY(student_id)   REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS posts (
        id           TEXT PRIMARY KEY,
        classroom_id TEXT NOT NULL,
        user_id      TEXT NOT NULL,
        title        TEXT NOT NULL,
        content      TEXT DEFAULT '',
        type         TEXT DEFAULT 'announcement',
        file_url     TEXT DEFAULT '',
        created_at   TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(classroom_id) REFERENCES classrooms(id),
        FOREIGN KEY(user_id)      REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS comments (
        id         TEXT PRIMARY KEY,
        post_id    TEXT NOT NULL,
        user_id    TEXT NOT NULL,
        comment    TEXT NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(post_id)  REFERENCES posts(id),
        FOREIGN KEY(user_id)  REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS assignments (
        id           TEXT PRIMARY KEY,
        classroom_id TEXT NOT NULL,
        title        TEXT NOT NULL,
        description  TEXT DEFAULT '',
        due_date     TEXT NOT NULL,
        file_url     TEXT DEFAULT '',
        created_at   TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(classroom_id) REFERENCES classrooms(id)
    );
    CREATE TABLE IF NOT EXISTS submissions (
        id            TEXT PRIMARY KEY,
        assignment_id TEXT NOT NULL,
        student_id    TEXT NOT NULL,
        file_url      TEXT DEFAULT '',
        status        TEXT DEFAULT 'submitted',
        submitted_at  TEXT DEFAULT (datetime('now')),
        UNIQUE(assignment_id, student_id),
        FOREIGN KEY(assignment_id) REFERENCES assignments(id),
        FOREIGN KEY(student_id)    REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS notifications (
        id         TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL,
        title      TEXT NOT NULL,
        message    TEXT NOT NULL,
        type       TEXT DEFAULT 'info',
        read       INTEGER DEFAULT 0,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    CREATE TABLE IF NOT EXISTS face_audit (
        id              TEXT PRIMARY KEY,
        classroom_id    TEXT NOT NULL,
        student_id      TEXT NOT NULL,
        action          TEXT NOT NULL,
        performed_by    TEXT NOT NULL,
        notes           TEXT DEFAULT '',
        performed_at    TEXT DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS otps (
        email      TEXT PRIMARY KEY,
        otp        TEXT NOT NULL,
        expires_at TEXT NOT NULL
    );
    """)
    db.commit()
    db.close()
    log.info("Database ready.")

init_db()

# ─────────────────────────── AUTH HELPERS ────────────────────────────────────
def hash_pwd(p): return pwd_ctx.hash(p)
def verify_pwd(plain, hashed): return pwd_ctx.verify(plain, hashed)

def create_token(uid: str):
    exp = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode({"sub": uid, "exp": exp}, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None

def uid(): return str(uuid.uuid4())

def gen_code(length=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def now_str(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def today():   return datetime.now().strftime("%Y-%m-%d")
def timeof():  return datetime.now().strftime("%H:%M:%S")

def get_current_user(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth.split(" ", 1)[1]
    uid_val = decode_token(token)
    if not uid_val:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id=?", (uid_val,)).fetchone()
    db.close()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return dict(user)

def send_notif(user_id: str, title: str, message: str, ntype: str = "info"):
    try:
        db = get_db()
        db.execute(
            "INSERT INTO notifications(id,user_id,title,message,type) VALUES(?,?,?,?,?)",
            (uid(), user_id, title, message, ntype)
        )
        db.commit()
        db.close()
    except Exception as e:
        log.error(f"Notif error: {e}")

# ─────────────────────────── FACE ENGINE ─────────────────────────────────────
def decode_b64_image(b64: str):
    """Decode base64 image to numpy array (RGB)."""
    if not b64:
        return None
    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        if CV2_OK:
            img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                return None
            return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            from PIL import Image
            import io
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            return np.array(pil)
    except Exception as e:
        log.error(f"Image decode error: {e}")
        return None

def get_face_encoding(img_rgb):
    """Get 128-d face encoding. Returns list or None."""
    if FACE_LIB == "face_recognition" and fr:
        try:
            encs = fr.face_encodings(img_rgb)
            if encs:
                return encs[0].tolist()
            return None
        except Exception as e:
            log.error(f"face_recognition error: {e}")
            return None
    elif CV2_OK:
        return _opencv_encoding(img_rgb)
    return None

def _opencv_encoding(img_rgb):
    """OpenCV HOG-based face detection + simple pixel encoding fallback."""
    try:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(60, 60))
        if len(faces) == 0:
            return None
        # Largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_crop = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
        flat = face_crop.flatten().astype(np.float32)
        norm = flat / (np.linalg.norm(flat) + 1e-8)
        return norm.tolist()
    except Exception as e:
        log.error(f"OpenCV encoding error: {e}")
        return None

def face_distance(enc1: list, enc2: list) -> float:
    """Euclidean distance between two encodings."""
    a, b = np.array(enc1), np.array(enc2)
    return float(np.linalg.norm(a - b))

def face_similarity_pct(enc1: list, enc2: list) -> float:
    """Return similarity as percentage (0–100)."""
    dist = face_distance(enc1, enc2)
    if FACE_LIB == "face_recognition":
        # face_recognition threshold ~0.6
        sim = max(0.0, 1.0 - dist / 0.8)
    else:
        # OpenCV encoding — normalised vectors, typical dist 0–2
        sim = max(0.0, 1.0 - dist / 1.5)
    return round(sim * 100, 2)

MATCH_THRESHOLD_PCT = 60.0   # min similarity to mark present

# ─────────────────────────── PYDANTIC MODELS ─────────────────────────────────
class RegisterIn(BaseModel):
    name: str
    email: str
    password: str
    department: str = ""

class LoginIn(BaseModel):
    email: str
    password: str

class CreateClassroomIn(BaseModel):
    name: str
    subject: str
    branch: str
    year: int
    section: str
    description: str = ""
    banner_color: str = "#1255a6"

class JoinClassroomIn(BaseModel):
    code: str
    roll_number: str
    branch: str = "CSE"
    year: int = 1
    section: str = "A"
    image: str = ""

class PostIn(BaseModel):
    classroom_id: str
    title: str
    content: str = ""
    type: str = "announcement"

class CommentIn(BaseModel):
    post_id: str
    comment: str

class DueDateIn(BaseModel):
    assignment_id: str
    due_date: str

class RecognizeIn(BaseModel):
    classroom_id: str
    image: str

class ForgotIn(BaseModel):
    email: str

class VerifyOTPIn(BaseModel):
    email: str
    otp: str
    new_password: str

class AdminResetFaceIn(BaseModel):
    classroom_id: str
    student_id: str
    image: str
    notes: str = "Admin reset"

# ─────────────────────────── AUTH ROUTES ─────────────────────────────────────
@app.post("/register")
def register(body: RegisterIn):
    db = get_db()
    if db.execute("SELECT id FROM users WHERE email=?", (body.email.lower().strip(),)).fetchone():
        db.close()
        raise HTTPException(400, "Email already registered.")
    if len(body.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    user_id = uid()
    db.execute(
        "INSERT INTO users(id,name,email,password,department) VALUES(?,?,?,?,?)",
        (user_id, body.name.strip(), body.email.lower().strip(),
         hash_pwd(body.password), body.department)
    )
    db.commit()
    db.close()
    send_notif(user_id, "Welcome to NITJ Classroom! 🎉",
               f"Hello {body.name}! Your account is ready.", "success")
    return {"token": create_token(user_id), "message": "Account created successfully!"}

@app.post("/login")
def login(body: LoginIn):
    db = get_db()
    user = db.execute(
        "SELECT * FROM users WHERE email=?", (body.email.lower().strip(),)
    ).fetchone()
    db.close()
    if not user or not verify_pwd(body.password, user["password"]):
        raise HTTPException(401, "Invalid email or password.")
    return {"token": create_token(user["id"]), "message": f"Welcome back, {user['name']}!"}

@app.post("/forgot_password")
def forgot_password(body: ForgotIn):
    db = get_db()
    user = db.execute("SELECT id FROM users WHERE email=?",
                      (body.email.lower().strip(),)).fetchone()
    if not user:
        raise HTTPException(404, "No account with that email.")
    otp = ''.join(random.choices(string.digits, k=6))
    expires = (datetime.now() + timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M:%S")
    db.execute("INSERT OR REPLACE INTO otps(email,otp,expires_at) VALUES(?,?,?)",
               (body.email.lower().strip(), otp, expires))
    db.commit()
    db.close()
    # In production: send via email (SMTP/SendGrid)
    log.info(f"OTP for {body.email}: {otp}")
    return {"message": "OTP sent to your email.", "otp": otp}  # remove otp in production!

@app.post("/verify_otp")
def verify_otp(body: VerifyOTPIn):
    db = get_db()
    row = db.execute("SELECT * FROM otps WHERE email=?",
                     (body.email.lower().strip(),)).fetchone()
    if not row:
        raise HTTPException(400, "No OTP found. Request a new one.")
    if row["otp"] != body.otp:
        raise HTTPException(400, "Incorrect OTP.")
    if datetime.now() > datetime.strptime(row["expires_at"], "%Y-%m-%d %H:%M:%S"):
        raise HTTPException(400, "OTP expired. Request a new one.")
    if len(body.new_password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters.")
    db.execute("UPDATE users SET password=? WHERE email=?",
               (hash_pwd(body.new_password), body.email.lower().strip()))
    db.execute("DELETE FROM otps WHERE email=?", (body.email.lower().strip(),))
    db.commit()
    db.close()
    return {"message": "Password reset successfully!"}

@app.get("/me")
def me(user=Depends(get_current_user)):
    return {k: v for k, v in user.items() if k != "password"}

# ─────────────────────────── CLASSROOMS ──────────────────────────────────────
@app.post("/create_classroom")
def create_classroom(body: CreateClassroomIn, user=Depends(get_current_user)):
    db = get_db()
    # Unique code
    for _ in range(20):
        code = gen_code()
        if not db.execute("SELECT id FROM classrooms WHERE code=?", (code,)).fetchone():
            break
    cls_id = uid()
    db.execute(
        """INSERT INTO classrooms(id,name,subject,branch,year,section,description,banner_color,code,admin_id)
           VALUES(?,?,?,?,?,?,?,?,?,?)""",
        (cls_id, body.name, body.subject, body.branch, body.year,
         body.section, body.description, body.banner_color, code, user["id"])
    )
    # Admin auto-enrolled
    db.execute(
        "INSERT INTO enrollments(id,classroom_id,user_id,roll_number) VALUES(?,?,?,?)",
        (uid(), cls_id, user["id"], "ADMIN")
    )
    db.commit()
    db.close()
    send_notif(user["id"], f"Classroom '{body.name}' created!",
               f"Share code {code} with students.", "success")
    return {"message": "Classroom created!", "code": code, "id": cls_id}

@app.get("/classrooms")
def list_classrooms(user=Depends(get_current_user)):
    db = get_db()
    rows = db.execute("""
        SELECT c.*, 
               (c.admin_id = ?) AS is_admin,
               (SELECT COUNT(*) FROM enrollments e2 WHERE e2.classroom_id=c.id) AS member_count,
               (SELECT COUNT(*) FROM assignments a WHERE a.classroom_id=c.id
                  AND a.due_date >= date('now')) AS upcoming_assignments
        FROM classrooms c
        JOIN enrollments e ON e.classroom_id=c.id
        WHERE e.user_id=?
        ORDER BY c.created_at DESC
    """, (user["id"], user["id"])).fetchall()
    db.close()
    return [dict(r) for r in rows]

@app.get("/classroom/{cls_id}")
def get_classroom(cls_id: str, user=Depends(get_current_user)):
    db = get_db()
    row = db.execute("""
        SELECT c.*,
               (c.admin_id = ?) AS is_admin,
               (SELECT COUNT(*) FROM enrollments e2 WHERE e2.classroom_id=c.id) AS member_count
        FROM classrooms c
        JOIN enrollments e ON e.classroom_id=c.id AND e.user_id=?
        WHERE c.id=?
    """, (user["id"], user["id"], cls_id)).fetchone()
    if not row:
        raise HTTPException(404, "Classroom not found or not enrolled.")
    enr = db.execute(
        "SELECT face_enrolled FROM enrollments WHERE classroom_id=? AND user_id=?",
        (cls_id, user["id"])
    ).fetchone()
    db.close()
    data = dict(row)
    data["face_enrolled"] = bool(enr and enr["face_enrolled"])
    return data

@app.delete("/classroom/{cls_id}")
def delete_classroom(cls_id: str, user=Depends(get_current_user)):
    db = get_db()
    cls = db.execute("SELECT * FROM classrooms WHERE id=? AND admin_id=?",
                     (cls_id, user["id"])).fetchone()
    if not cls:
        raise HTTPException(403, "Not authorized or classroom not found.")
    db.execute("DELETE FROM classrooms WHERE id=?", (cls_id,))
    db.execute("DELETE FROM enrollments WHERE classroom_id=?", (cls_id,))
    db.execute("DELETE FROM attendance WHERE classroom_id=?", (cls_id,))
    db.execute("DELETE FROM posts WHERE classroom_id=?", (cls_id,))
    db.execute("DELETE FROM assignments WHERE classroom_id=?", (cls_id,))
    db.commit()
    db.close()
    return {"message": "Classroom deleted."}

@app.post("/join_classroom")
def join_classroom(body: JoinClassroomIn, user=Depends(get_current_user)):
    db = get_db()
    cls = db.execute("SELECT * FROM classrooms WHERE code=?",
                     (body.code.upper().strip(),)).fetchone()
    if not cls:
        raise HTTPException(404, "Invalid classroom code.")
    if cls["admin_id"] == user["id"]:
        raise HTTPException(400, "You are the admin of this classroom.")
    existing = db.execute(
        "SELECT id FROM enrollments WHERE classroom_id=? AND user_id=?",
        (cls["id"], user["id"])
    ).fetchone()
    if existing:
        raise HTTPException(400, "Already enrolled in this classroom.")

    # Face encoding
    face_data = ""
    face_enrolled = 0
    if body.image:
        img = decode_b64_image(body.image)
        if img is not None:
            enc = get_face_encoding(img)
            if enc:
                face_data = json.dumps(enc)
                face_enrolled = 1
            else:
                log.info("No face detected in join image.")

    enr_id = uid()
    db.execute(
        """INSERT INTO enrollments(id,classroom_id,user_id,roll_number,branch,year,section,face_data,face_enrolled)
           VALUES(?,?,?,?,?,?,?,?,?)""",
        (enr_id, cls["id"], user["id"], body.roll_number,
         body.branch, body.year, body.section, face_data, face_enrolled)
    )
    db.commit()

    # Notify admin
    admin = db.execute("SELECT name FROM users WHERE id=?", (cls["admin_id"],)).fetchone()
    send_notif(cls["admin_id"],
               f"New student joined '{cls['name']}'",
               f"{user['name']} (Roll: {body.roll_number}) joined.", "info")
    db.close()

    # Face audit
    if face_enrolled:
        db2 = get_db()
        db2.execute(
            "INSERT INTO face_audit(id,classroom_id,student_id,action,performed_by,notes) VALUES(?,?,?,?,?,?)",
            (uid(), cls["id"], user["id"], "ENROLLED", user["id"], "Self-enrolled on join")
        )
        db2.commit()
        db2.close()

    msg = "Joined successfully!"
    if not face_enrolled and body.image:
        msg += " (No face detected — contact teacher to enroll face.)"
    return {"message": msg, "face_enrolled": face_enrolled}

@app.get("/classroom/{cls_id}/members")
def get_members(cls_id: str, user=Depends(get_current_user)):
    db = get_db()
    # Must be enrolled
    if not db.execute("SELECT id FROM enrollments WHERE classroom_id=? AND user_id=?",
                      (cls_id, user["id"])).fetchone():
        raise HTTPException(403, "Not enrolled.")
    cls = db.execute("SELECT admin_id FROM classrooms WHERE id=?", (cls_id,)).fetchone()
    rows = db.execute("""
        SELECT u.id, u.name, u.email, u.department,
               e.roll_number, e.branch, e.year, e.section, e.face_enrolled,
               (u.id = c.admin_id) AS is_admin
        FROM enrollments e
        JOIN users u ON u.id = e.user_id
        JOIN classrooms c ON c.id = e.classroom_id
        WHERE e.classroom_id=?
        ORDER BY is_admin DESC, u.name
    """, (cls_id,)).fetchall()
    db.close()
    return [dict(r) for r in rows]

@app.delete("/classroom/{cls_id}/remove/{student_id}")
def remove_member(cls_id: str, student_id: str, user=Depends(get_current_user)):
    db = get_db()
    cls = db.execute("SELECT admin_id FROM classrooms WHERE id=?", (cls_id,)).fetchone()
    if not cls or cls["admin_id"] != user["id"]:
        raise HTTPException(403, "Only admin can remove members.")
    if student_id == user["id"]:
        raise HTTPException(400, "Cannot remove yourself.")
    db.execute("DELETE FROM enrollments WHERE classroom_id=? AND user_id=?",
               (cls_id, student_id))
    db.commit()
    db.close()
    return {"message": "Member removed."}

# ─────────────────────────── POSTS ───────────────────────────────────────────
@app.post("/post")
def create_post(body: PostIn, user=Depends(get_current_user)):
    db = get_db()
    cls = db.execute("SELECT admin_id FROM classrooms WHERE id=?",
                     (body.classroom_id,)).fetchone()
    if not cls or cls["admin_id"] != user["id"]:
        raise HTTPException(403, "Only admin can post.")
    post_id = uid()
    db.execute(
        "INSERT INTO posts(id,classroom_id,user_id,title,content,type) VALUES(?,?,?,?,?,?)",
        (post_id, body.classroom_id, user["id"], body.title, body.content, body.type)
    )
    # Notify all students
    students = db.execute(
        "SELECT user_id FROM enrollments WHERE classroom_id=? AND user_id != ?",
        (body.classroom_id, user["id"])
    ).fetchall()
    db.commit()
    db.close()
    for s in students:
        send_notif(s["user_id"], f"📢 New post: {body.title}",
                   f"{body.content[:100]}", "info")
    return {"message": "Posted!", "id": post_id}

@app.get("/posts/{cls_id}")
def get_posts(cls_id: str, user=Depends(get_current_user)):
    db = get_db()
    rows = db.execute("""
        SELECT p.*, u.name AS user_name,
               (SELECT COUNT(*) FROM comments c WHERE c.post_id=p.id) AS comment_count
        FROM posts p
        JOIN users u ON u.id=p.user_id
        WHERE p.classroom_id=?
        ORDER BY p.created_at DESC
    """, (cls_id,)).fetchall()
    db.close()
    return [dict(r) for r in rows]

@app.delete("/post/{post_id}")
def delete_post(post_id: str, user=Depends(get_current_user)):
    db = get_db()
    post = db.execute("SELECT * FROM posts WHERE id=?", (post_id,)).fetchone()
    if not post:
        raise HTTPException(404, "Post not found.")
    cls = db.execute("SELECT admin_id FROM classrooms WHERE id=?",
                     (post["classroom_id"],)).fetchone()
    if not cls or cls["admin_id"] != user["id"]:
        raise HTTPException(403, "Only admin can delete posts.")
    db.execute("DELETE FROM posts WHERE id=?", (post_id,))
    db.execute("DELETE FROM comments WHERE post_id=?", (post_id,))
    db.commit()
    db.close()
    return {"message": "Deleted."}

@app.post("/post_with_file")
async def post_with_file(
    classroom_id: str = Form(...),
    title: str = Form(...),
    content: str = Form(""),
    type: str = Form("material"),
    file: UploadFile = File(None),
    user=Depends(get_current_user)
):
    db = get_db()
    cls = db.execute("SELECT admin_id FROM classrooms WHERE id=?", (classroom_id,)).fetchone()
    if not cls or cls["admin_id"] != user["id"]:
        raise HTTPException(403, "Only admin can post.")
    file_url = ""
    if file and file.filename:
        ext = Path(file.filename).suffix
        fname = f"post_{uid()}{ext}"
        fpath = UPLOAD_DIR / fname
        with open(fpath, "wb") as f:
            f.write(await file.read())
        file_url = f"/uploads/{fname}"
    post_id = uid()
    db.execute(
        "INSERT INTO posts(id,classroom_id,user_id,title,content,type,file_url) VALUES(?,?,?,?,?,?,?)",
        (post_id, classroom_id, user["id"], title, content, type, file_url)
    )
    db.commit()
    db.close()
    return {"message": "Posted!", "id": post_id}

# ─────────────────────────── COMMENTS ────────────────────────────────────────
@app.post("/comment")
def add_comment(body: CommentIn, user=Depends(get_current_user)):
    db = get_db()
    db.execute(
        "INSERT INTO comments(id,post_id,user_id,comment) VALUES(?,?,?,?)",
        (uid(), body.post_id, user["id"], body.comment.strip())
    )
    db.commit()
    db.close()
    return {"message": "Comment added."}

@app.get("/comments/{post_id}")
def get_comments(post_id: str, user=Depends(get_current_user)):
    db = get_db()
    rows = db.execute("""
        SELECT c.*, u.name AS user_name
        FROM comments c
        JOIN users u ON u.id=c.user_id
        WHERE c.post_id=?
        ORDER BY c.created_at
    """, (post_id,)).fetchall()
    db.close()
    return [dict(r) for r in rows]

# ─────────────────────────── ASSIGNMENTS ─────────────────────────────────────
@app.post("/create_assignment")
async def create_assignment(
    classroom_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(""),
    due_date: str = Form(...),
    file: UploadFile = File(None),
    user=Depends(get_current_user)
):
    db = get_db()
    cls = db.execute("SELECT admin_id, name FROM classrooms WHERE id=?", (classroom_id,)).fetchone()
    if not cls or cls["admin_id"] != user["id"]:
        raise HTTPException(403, "Only admin can create assignments.")
    file_url = ""
    if file and file.filename:
        ext = Path(file.filename).suffix
        fname = f"asgn_{uid()}{ext}"
        fpath = UPLOAD_DIR / fname
        with open(fpath, "wb") as f:
            f.write(await file.read())
        file_url = f"/uploads/{fname}"
    asgn_id = uid()
    db.execute(
        "INSERT INTO assignments(id,classroom_id,title,description,due_date,file_url) VALUES(?,?,?,?,?,?)",
        (asgn_id, classroom_id, title, description, due_date, file_url)
    )
    students = db.execute(
        "SELECT user_id FROM enrollments WHERE classroom_id=? AND user_id != ?",
        (classroom_id, user["id"])
    ).fetchall()
    db.commit()
    db.close()
    for s in students:
        send_notif(s["user_id"], f"📋 New assignment: {title}",
                   f"Due: {due_date}. Class: {cls['name']}", "warn")
    return {"message": "Assignment created!", "id": asgn_id}

@app.get("/assignments/{cls_id}")
def get_assignments(cls_id: str, user=Depends(get_current_user)):
    db = get_db()
    cls = db.execute("SELECT admin_id FROM classrooms WHERE id=?", (cls_id,)).fetchone()
    is_admin = cls and cls["admin_id"] == user["id"]
    rows = db.execute("""
        SELECT a.*,
               (a.due_date < date('now')) AS is_overdue,
               (SELECT COUNT(*) FROM submissions s WHERE s.assignment_id=a.id) AS submission_count
        FROM assignments a
        WHERE a.classroom_id=?
        ORDER BY a.due_date ASC
    """, (cls_id,)).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        # Student's own submission
        sub = db.execute(
            "SELECT * FROM submissions WHERE assignment_id=? AND student_id=?",
            (r["id"], user["id"])
        ).fetchone()
        d["my_submission"] = dict(sub) if sub else None
        result.append(d)
    db.close()
    return result

@app.get("/my_assignments")
def my_assignments(user=Depends(get_current_user)):
    db = get_db()
    rows = db.execute("""
        SELECT a.*, c.name AS classroom_name, c.subject,
               (a.due_date < date('now')) AS is_overdue,
               (SELECT COUNT(*) FROM submissions s WHERE s.assignment_id=a.id) AS submission_count
        FROM assignments a
        JOIN classrooms c ON c.id=a.classroom_id
        JOIN enrollments e ON e.classroom_id=c.id AND e.user_id=?
        ORDER BY a.due_date ASC
    """, (user["id"],)).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        sub = db.execute(
            "SELECT * FROM submissions WHERE assignment_id=? AND student_id=?",
            (r["id"], user["id"])
        ).fetchone()
        d["my_submission"] = dict(sub) if sub else None
        result.append(d)
    db.close()
    return result

@app.put("/assignment/{asgn_id}/due_date")
def update_due_date(asgn_id: str, body: DueDateIn, user=Depends(get_current_user)):
    db = get_db()
    asgn = db.execute(
        "SELECT a.*, c.admin_id FROM assignments a JOIN classrooms c ON c.id=a.classroom_id WHERE a.id=?",
        (asgn_id,)
    ).fetchone()
    if not asgn:
        raise HTTPException(404, "Assignment not found.")
    if asgn["admin_id"] != user["id"]:
        raise HTTPException(403, "Only admin can update.")
    db.execute("UPDATE assignments SET due_date=? WHERE id=?", (body.due_date, asgn_id))
    db.commit()
    db.close()
    return {"message": "Due date updated."}

@app.delete("/assignment/{asgn_id}")
def delete_assignment(asgn_id: str, user=Depends(get_current_user)):
    db = get_db()
    asgn = db.execute(
        "SELECT a.*, c.admin_id FROM assignments a JOIN classrooms c ON c.id=a.classroom_id WHERE a.id=?",
        (asgn_id,)
    ).fetchone()
    if not asgn or asgn["admin_id"] != user["id"]:
        raise HTTPException(403, "Not authorized.")
    db.execute("DELETE FROM assignments WHERE id=?", (asgn_id,))
    db.execute("DELETE FROM submissions WHERE assignment_id=?", (asgn_id,))
    db.commit()
    db.close()
    return {"message": "Deleted."}

@app.post("/submit_assignment")
async def submit_assignment(
    assignment_id: str = Form(...),
    file: UploadFile = File(None),
    user=Depends(get_current_user)
):
    db = get_db()
    asgn = db.execute("SELECT * FROM assignments WHERE id=?", (assignment_id,)).fetchone()
    if not asgn:
        raise HTTPException(404, "Assignment not found.")
    file_url = ""
    if file and file.filename:
        ext = Path(file.filename).suffix
        fname = f"sub_{uid()}{ext}"
        fpath = UPLOAD_DIR / fname
        with open(fpath, "wb") as f:
            f.write(await file.read())
        file_url = f"/uploads/{fname}"
    is_late = asgn["due_date"] < today()
    status_val = "late" if is_late else "submitted"
    existing = db.execute(
        "SELECT id FROM submissions WHERE assignment_id=? AND student_id=?",
        (assignment_id, user["id"])
    ).fetchone()
    if existing:
        db.execute(
            "UPDATE submissions SET file_url=?,status=?,submitted_at=? WHERE id=?",
            (file_url, status_val, now_str(), existing["id"])
        )
    else:
        db.execute(
            "INSERT INTO submissions(id,assignment_id,student_id,file_url,status) VALUES(?,?,?,?,?)",
            (uid(), assignment_id, user["id"], file_url, status_val)
        )
    db.commit()
    db.close()
    msg = "Submitted late!" if is_late else "Submitted successfully!"
    return {"message": msg, "status": status_val}

@app.get("/submissions/{asgn_id}")
def get_submissions(asgn_id: str, user=Depends(get_current_user)):
    db = get_db()
    asgn = db.execute(
        "SELECT a.*, c.admin_id FROM assignments a JOIN classrooms c ON c.id=a.classroom_id WHERE a.id=?",
        (asgn_id,)
    ).fetchone()
    if not asgn or asgn["admin_id"] != user["id"]:
        raise HTTPException(403, "Only admin can view submissions.")
    rows = db.execute("""
        SELECT s.*, u.name AS student_name, u.email
        FROM submissions s
        JOIN users u ON u.id=s.student_id
        WHERE s.assignment_id=?
        ORDER BY s.submitted_at DESC
    """, (asgn_id,)).fetchall()
    db.close()
    return [dict(r) for r in rows]

# ─────────────────────────── MATERIAL UPLOAD ─────────────────────────────────
@app.post("/upload_material")
async def upload_material(
    classroom_id: str = Form(...),
    title: str = Form(...),
    content: str = Form(""),
    file: UploadFile = File(None),
    user=Depends(get_current_user)
):
    db = get_db()
    cls = db.execute("SELECT admin_id, name FROM classrooms WHERE id=?", (classroom_id,)).fetchone()
    if not cls or cls["admin_id"] != user["id"]:
        raise HTTPException(403, "Only admin can upload.")
    file_url = ""
    if file and file.filename:
        ext = Path(file.filename).suffix
        fname = f"mat_{uid()}{ext}"
        fpath = UPLOAD_DIR / fname
        with open(fpath, "wb") as f:
            f.write(await file.read())
        file_url = f"/uploads/{fname}"
    post_id = uid()
    db.execute(
        "INSERT INTO posts(id,classroom_id,user_id,title,content,type,file_url) VALUES(?,?,?,?,?,?,?)",
        (post_id, classroom_id, user["id"], title, content, "material", file_url)
    )
    students = db.execute(
        "SELECT user_id FROM enrollments WHERE classroom_id=? AND user_id != ?",
        (classroom_id, user["id"])
    ).fetchall()
    db.commit()
    db.close()
    for s in students:
        send_notif(s["user_id"], f"📄 New material: {title}",
                   f"New study material uploaded in {cls['name']}.", "info")
    return {"message": "Material uploaded!", "id": post_id}

# ─────────────────────────── ATTENDANCE ──────────────────────────────────────
@app.post("/recognize")
def recognize(body: RecognizeIn, user=Depends(get_current_user)):
    """AI face recognition for attendance."""
    db = get_db()
    # Must be admin
    cls = db.execute("SELECT admin_id FROM classrooms WHERE id=?", (body.classroom_id,)).fetchone()
    if not cls or cls["admin_id"] != user["id"]:
        raise HTTPException(403, "Only admin can take attendance.")

    # Decode frame
    img = decode_b64_image(body.image)
    if img is None:
        return {"status": "error", "results": [], "message": "Could not decode image."}

    # Detect faces in frame
    if FACE_LIB == "face_recognition" and fr:
        try:
            face_locs = fr.face_locations(img, model="hog")
            frame_encs = fr.face_encodings(img, face_locs)
        except Exception as e:
            return {"status": "error", "results": [], "message": str(e)}
    elif CV2_OK:
        # Use OpenCV to get all face encodings
        single_enc = _opencv_encoding(img)
        frame_encs = [single_enc] if single_enc else []
    else:
        return {"status": "error", "results": [], "message": "No face recognition library."}

    if not frame_encs:
        return {"status": "no_face", "results": [], "message": "No face detected in frame."}

    # Load enrolled students
    enrollments = db.execute("""
        SELECT e.user_id, e.face_data, e.roll_number,
               u.name
        FROM enrollments e
        JOIN users u ON u.id=e.user_id
        WHERE e.classroom_id=? AND e.face_enrolled=1 AND e.user_id != ?
    """, (body.classroom_id, user["id"])).fetchall()

    if not enrollments:
        db.close()
        return {"status": "no_students", "results": [],
                "message": "No students with face data enrolled."}

    results = []
    cur_date = today()
    cur_time = timeof()

    for frame_enc in frame_encs:
        if frame_enc is None:
            continue
        best_match = None
        best_sim = 0.0

        for enr in enrollments:
            if not enr["face_data"]:
                continue
            try:
                stored_enc = json.loads(enr["face_data"])
                sim = face_similarity_pct(frame_enc, stored_enc)
                if sim > best_sim:
                    best_sim = sim
                    best_match = enr
            except Exception as e:
                log.error(f"Encoding compare error: {e}")
                continue

        if best_match and best_sim >= MATCH_THRESHOLD_PCT:
            # Check duplicate for today
            already = db.execute(
                "SELECT id FROM attendance WHERE classroom_id=? AND student_id=? AND date=? AND status='present'",
                (body.classroom_id, best_match["user_id"], cur_date)
            ).fetchone()
            if already:
                results.append({
                    "status": "duplicate",
                    "name": best_match["name"],
                    "roll": best_match["roll_number"],
                    "confidence": best_sim
                })
            else:
                # Mark present
                db.execute(
                    "INSERT INTO attendance(id,classroom_id,student_id,date,time,status,confidence) VALUES(?,?,?,?,?,?,?)",
                    (uid(), body.classroom_id, best_match["user_id"],
                     cur_date, cur_time, "present", best_sim)
                )
                db.commit()
                results.append({
                    "status": "present",
                    "name": best_match["name"],
                    "roll": best_match["roll_number"],
                    "confidence": best_sim
                })
                send_notif(best_match["user_id"],
                           "✅ Attendance Marked",
                           f"Present on {cur_date} at {cur_time[:5]}. Confidence: {best_sim:.1f}%",
                           "success")
        else:
            results.append({"status": "unknown", "name": "Unknown",
                            "confidence": best_sim})

    db.close()
    return {"status": "ok", "results": results}

@app.post("/mark_absent/{cls_id}")
def mark_absent(cls_id: str, user=Depends(get_current_user)):
    """Mark all un-marked students as absent for today."""
    db = get_db()
    cls = db.execute("SELECT admin_id FROM classrooms WHERE id=?", (cls_id,)).fetchone()
    if not cls or cls["admin_id"] != user["id"]:
        raise HTTPException(403, "Only admin can do this.")
    cur_date = today()
    students = db.execute(
        "SELECT user_id FROM enrollments WHERE classroom_id=? AND user_id != ?",
        (cls_id, user["id"])
    ).fetchall()
    count = 0
    for s in students:
        already = db.execute(
            "SELECT id FROM attendance WHERE classroom_id=? AND student_id=? AND date=?",
            (cls_id, s["user_id"], cur_date)
        ).fetchone()
        if not already:
            db.execute(
                "INSERT INTO attendance(id,classroom_id,student_id,date,time,status,confidence) VALUES(?,?,?,?,?,?,?)",
                (uid(), cls_id, s["user_id"], cur_date, timeof(), "absent", 0)
            )
            count += 1
    db.commit()
    db.close()
    return {"message": f"{count} students marked absent."}

@app.get("/attendance/{cls_id}")
def get_attendance(cls_id: str, date_filter: str = None, user=Depends(get_current_user)):
    db = get_db()
    # Verify enrolled
    if not db.execute("SELECT id FROM enrollments WHERE classroom_id=? AND user_id=?",
                      (cls_id, user["id"])).fetchone():
        raise HTTPException(403, "Not enrolled.")
    cls = db.execute("SELECT admin_id FROM classrooms WHERE id=?", (cls_id,)).fetchone()
    is_admin = cls and cls["admin_id"] == user["id"]

    query = """
        SELECT a.*, u.name AS student_name
        FROM attendance a
        JOIN users u ON u.id=a.student_id
        WHERE a.classroom_id=?
    """
    params = [cls_id]
    if not is_admin:
        query += " AND a.student_id=?"
        params.append(user["id"])
    if date_filter:
        query += " AND a.date=?"
        params.append(date_filter)
    query += " ORDER BY a.date DESC, a.time DESC"
    rows = db.execute(query, params).fetchall()
    db.close()
    return [dict(r) for r in rows]

@app.get("/my_attendance")
def my_attendance(user=Depends(get_current_user)):
    db = get_db()
    rows = db.execute("""
        SELECT a.*, c.name AS classroom_name, c.subject
        FROM attendance a
        JOIN classrooms c ON c.id=a.classroom_id
        WHERE a.student_id=?
        ORDER BY a.date DESC
    """, (user["id"],)).fetchall()
    db.close()
    return [dict(r) for r in rows]

# ─────────────────────────── FACE MANAGEMENT ─────────────────────────────────
@app.post("/admin_reset_face")
def admin_reset_face(body: AdminResetFaceIn, user=Depends(get_current_user)):
    db = get_db()
    cls = db.execute("SELECT admin_id FROM classrooms WHERE id=?", (body.classroom_id,)).fetchone()
    if not cls or cls["admin_id"] != user["id"]:
        raise HTTPException(403, "Only admin.")
    img = decode_b64_image(body.image)
    if img is None:
        raise HTTPException(400, "Bad image.")
    enc = get_face_encoding(img)
    if not enc:
        raise HTTPException(400, "No face detected in image. Try again.")
    db.execute(
        "UPDATE enrollments SET face_data=?,face_enrolled=1 WHERE classroom_id=? AND user_id=?",
        (json.dumps(enc), body.classroom_id, body.student_id)
    )
    db.execute(
        "INSERT INTO face_audit(id,classroom_id,student_id,action,performed_by,notes) VALUES(?,?,?,?,?,?)",
        (uid(), body.classroom_id, body.student_id, "ADMIN_RESET", user["id"], body.notes)
    )
    db.commit()
    stu = db.execute("SELECT name FROM users WHERE id=?", (body.student_id,)).fetchone()
    db.close()
    send_notif(body.student_id, "Face Data Updated",
               "Your face data was updated by the teacher.", "info")
    return {"message": f"Face updated for {stu['name'] if stu else body.student_id}."}

@app.post("/admin_clear_face")
async def admin_clear_face(
    classroom_id: str = Form(...),
    student_id: str = Form(...),
    notes: str = Form(""),
    user=Depends(get_current_user)
):
    db = get_db()
    cls = db.execute("SELECT admin_id FROM classrooms WHERE id=?", (classroom_id,)).fetchone()
    if not cls or cls["admin_id"] != user["id"]:
        raise HTTPException(403, "Only admin.")
    db.execute(
        "UPDATE enrollments SET face_data='',face_enrolled=0 WHERE classroom_id=? AND user_id=?",
        (classroom_id, student_id)
    )
    db.execute(
        "INSERT INTO face_audit(id,classroom_id,student_id,action,performed_by,notes) VALUES(?,?,?,?,?,?)",
        (uid(), classroom_id, student_id, "ADMIN_CLEARED", user["id"], notes or "Cleared by admin")
    )
    db.commit()
    stu = db.execute("SELECT name FROM users WHERE id=?", (student_id,)).fetchone()
    db.close()
    send_notif(student_id, "Face Data Cleared",
               "Your face data was cleared by the teacher. Contact them to re-enroll.", "warn")
    return {"message": f"Face cleared for {stu['name'] if stu else student_id}."}

@app.get("/admin/face_audit/{cls_id}")
def face_audit(cls_id: str, user=Depends(get_current_user)):
    db = get_db()
    cls = db.execute("SELECT admin_id FROM classrooms WHERE id=?", (cls_id,)).fetchone()
    if not cls or cls["admin_id"] != user["id"]:
        raise HTTPException(403, "Only admin.")
    rows = db.execute("""
        SELECT fa.*,
               us.name AS student_name,
               up.name AS performed_by_name
        FROM face_audit fa
        LEFT JOIN users us ON us.id=fa.student_id
        LEFT JOIN users up ON up.id=fa.performed_by
        WHERE fa.classroom_id=?
        ORDER BY fa.performed_at DESC
        LIMIT 50
    """, (cls_id,)).fetchall()
    db.close()
    return [dict(r) for r in rows]

# ─────────────────────────── NOTIFICATIONS ───────────────────────────────────
@app.get("/notifications")
def get_notifications(user=Depends(get_current_user)):
    db = get_db()
    rows = db.execute("""
        SELECT * FROM notifications WHERE user_id=?
        ORDER BY created_at DESC LIMIT 50
    """, (user["id"],)).fetchall()
    db.close()
    return [dict(r) for r in rows]

@app.post("/notifications/read_all")
def mark_all_read(user=Depends(get_current_user)):
    db = get_db()
    db.execute("UPDATE notifications SET read=1 WHERE user_id=?", (user["id"],))
    db.commit()
    db.close()
    return {"message": "All notifications marked as read."}

@app.patch("/notification/{notif_id}/read")
def mark_read(notif_id: str, user=Depends(get_current_user)):
    db = get_db()
    db.execute("UPDATE notifications SET read=1 WHERE id=? AND user_id=?",
               (notif_id, user["id"]))
    db.commit()
    db.close()
    return {"message": "Marked read."}

# ─────────────────────────── HEALTH ──────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    if FRONTEND_HTML.exists():
        return HTMLResponse(content=FRONTEND_HTML.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>NITJ Classroom API v9.0 — Place index.html in app folder</h2>")

@app.get("/health")
def health():
    return {"status": "ok", "time": now_str(), "face_engine": FACE_LIB}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
