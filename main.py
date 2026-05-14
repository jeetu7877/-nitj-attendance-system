
"""
NITJ Classroom Platform v9.0 - Complete Backend
Features: Face lock security, audit logs, notifications, assignments across classrooms,
          forgot password, face reset by teacher only

Install:
    pip install fastapi uvicorn[standard] python-multipart pillow numpy \
                face_recognition opencv-python-headless \
                python-jose[cryptography] passlib[bcrypt] aiofiles

Run:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import os, io, base64, pickle, sqlite3, logging, uuid, secrets, hashlib
from datetime import date, datetime, timedelta
from typing import Optional
import numpy as np
from PIL import Image

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt

DB_PATH    = "nitj.db"
PKL_PATH   = "faces.pkl"
UPLOAD_DIR = "uploads"
SECRET_KEY = os.getenv("SECRET_KEY", "nitj-2025-secret")
ALGORITHM  = "HS256"
TOKEN_EXP  = 60 * 24 * 7
TOLERANCE  = 0.48

os.makedirs(UPLOAD_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer  = HTTPBearer(auto_error=False)

app = FastAPI(title="NITJ Classroom API", version="9.0.0")

@app.exception_handler(RequestValidationError)
async def val_err(request, exc):
    msgs = [e["msg"] for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": "; ".join(msgs)})

app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

try:
    app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
except Exception:
    pass

try:
    import face_recognition as fr
    log.info("✅ face_recognition loaded")
except ImportError:
    fr = None
    log.error("❌ face_recognition not installed")

# ── DB ─────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            email           TEXT NOT NULL UNIQUE,
            hashed_password TEXT NOT NULL,
            department      TEXT DEFAULT '',
            created_at      TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS password_resets (
            id         TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            otp        TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used       INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS classrooms (
            id           TEXT PRIMARY KEY,
            creator_id   TEXT NOT NULL,
            name         TEXT NOT NULL,
            subject      TEXT NOT NULL,
            branch       TEXT NOT NULL,
            year         INTEGER NOT NULL,
            section      TEXT NOT NULL,
            code         TEXT NOT NULL UNIQUE,
            description  TEXT DEFAULT '',
            banner_color TEXT DEFAULT '#1565C0',
            created_at   TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (creator_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS classroom_members (
            id              TEXT PRIMARY KEY,
            classroom_id    TEXT NOT NULL,
            user_id         TEXT NOT NULL,
            roll_number     TEXT DEFAULT '',
            branch          TEXT DEFAULT '',
            year            INTEGER DEFAULT 1,
            section         TEXT DEFAULT '',
            face_enrolled   INTEGER DEFAULT 0,
            face_locked     INTEGER DEFAULT 0,
            face_updated_by TEXT DEFAULT '',
            face_updated_at TEXT DEFAULT '',
            is_admin        INTEGER DEFAULT 0,
            joined_at       TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (classroom_id) REFERENCES classrooms(id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(classroom_id, user_id)
        );
        CREATE TABLE IF NOT EXISTS face_audit_logs (
            id           TEXT PRIMARY KEY,
            classroom_id TEXT NOT NULL,
            student_id   TEXT NOT NULL,
            action       TEXT NOT NULL,
            performed_by TEXT NOT NULL,
            performed_at TEXT DEFAULT (datetime('now')),
            notes        TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id           TEXT PRIMARY KEY,
            classroom_id TEXT NOT NULL,
            student_id   TEXT NOT NULL,
            date         TEXT NOT NULL,
            time         TEXT NOT NULL,
            status       TEXT DEFAULT 'present',
            confidence   REAL DEFAULT 0,
            UNIQUE(classroom_id, student_id, date)
        );
        CREATE TABLE IF NOT EXISTS posts (
            id           TEXT PRIMARY KEY,
            classroom_id TEXT NOT NULL,
            user_id      TEXT NOT NULL,
            type         TEXT DEFAULT 'announcement',
            title        TEXT NOT NULL,
            content      TEXT DEFAULT '',
            file_url     TEXT DEFAULT '',
            file_name    TEXT DEFAULT '',
            created_at   TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS comments (
            id         TEXT PRIMARY KEY,
            post_id    TEXT NOT NULL,
            user_id    TEXT NOT NULL,
            comment    TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS assignments (
            id           TEXT PRIMARY KEY,
            classroom_id TEXT NOT NULL,
            creator_id   TEXT NOT NULL,
            title        TEXT NOT NULL,
            description  TEXT DEFAULT '',
            file_url     TEXT DEFAULT '',
            file_name    TEXT DEFAULT '',
            due_date     TEXT NOT NULL,
            created_at   TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS assignment_submissions (
            id            TEXT PRIMARY KEY,
            assignment_id TEXT NOT NULL,
            student_id    TEXT NOT NULL,
            file_url      TEXT DEFAULT '',
            file_name     TEXT DEFAULT '',
            submitted_at  TEXT DEFAULT (datetime('now')),
            status        TEXT DEFAULT 'submitted',
            UNIQUE(assignment_id, student_id)
        );
        CREATE TABLE IF NOT EXISTS notifications (
            id         TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            title      TEXT NOT NULL,
            message    TEXT NOT NULL,
            type       TEXT DEFAULT 'info',
            read       INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit(); conn.close()
    log.info("✅ DB ready: %s", DB_PATH)

init_db()

# ── Face store ─────────────────────────────────────────────────────
def load_faces():
    if os.path.exists(PKL_PATH):
        with open(PKL_PATH,"rb") as f: return pickle.load(f)
    return {}

def save_faces(store):
    with open(PKL_PATH,"wb") as f: pickle.dump(store,f)
    log.info("Saved faces.pkl — %d entries", len(store))

def b64_to_rgb(b64):
    if "," in b64: b64 = b64.split(",",1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    w,h = img.size
    if w > 800: img = img.resize((800,int(h*800/w)),Image.LANCZOS)
    return np.array(img)

def encode_face(rgb):
    if fr is None: raise ValueError("face_recognition not installed.")
    locs = fr.face_locations(rgb, model="hog")
    if not locs: raise ValueError("No face detected. Improve lighting.")
    if len(locs)>1: raise ValueError(f"{len(locs)} faces detected. Only 1 person please.")
    encs = fr.face_encodings(rgb, known_face_locations=locs, num_jitters=2)
    if not encs: raise ValueError("Could not generate encoding. Try again.")
    log.info("✅ Encoding generated shape=%s", encs[0].shape)
    return encs[0]

def send_notification(user_id, title, message, ntype="info"):
    conn = get_db()
    conn.execute("INSERT INTO notifications (id,user_id,title,message,type) VALUES (?,?,?,?,?)",
        (str(uuid.uuid4()), user_id, title, message, ntype))
    conn.commit(); conn.close()

# ── JWT ────────────────────────────────────────────────────────────
def create_token(uid):
    return jwt.encode({"sub":uid,"exp":datetime.utcnow()+timedelta(minutes=TOKEN_EXP)}, SECRET_KEY, algorithm=ALGORITHM)

def get_uid(creds: HTTPAuthorizationCredentials = Depends(bearer)):
    if not creds: raise HTTPException(401,"Not authenticated")
    try:
        p = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        uid = p.get("sub")
        if not uid: raise HTTPException(401,"Invalid token")
        return uid
    except JWTError: raise HTTPException(401,"Session expired. Please login again.")

def get_user(uid=Depends(get_uid)):
    conn = get_db()
    row  = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    if not row: raise HTTPException(401,"User not found")
    u = dict(row); u.pop("hashed_password",None)
    return u

def require_admin(classroom_id, user_id):
    conn = get_db()
    row  = conn.execute("SELECT is_admin FROM classroom_members WHERE classroom_id=? AND user_id=?",
        (classroom_id,user_id)).fetchone()
    conn.close()
    if not row or not row["is_admin"]: raise HTTPException(403,"Only classroom admin can do this.")
    return True

# ── Models ─────────────────────────────────────────────────────────
class RegReq(BaseModel):
    name:str; email:str; password:str; department:str=""

class LoginReq(BaseModel):
    email:str; password:str

class ForgotReq(BaseModel):
    email:str

class VerifyOTPReq(BaseModel):
    email:str; otp:str; new_password:str

class CreateClsReq(BaseModel):
    name:str; subject:str; branch:str; year:int
    section:str; description:str=""; banner_color:str="#1565C0"

class JoinClsReq(BaseModel):
    code:str; roll_number:str=""; branch:str=""
    year:int=1; section:str=""; image:str=""

class PostReq(BaseModel):
    classroom_id:str; type:str="announcement"; title:str; content:str=""

class CommentReq(BaseModel):
    post_id:str; comment:str

class RecognizeReq(BaseModel):
    classroom_id:str; image:str

class UpdateDueDateReq(BaseModel):
    assignment_id:str; due_date:str

class FaceResetReq(BaseModel):
    classroom_id:str; student_id:str; image:str; notes:str=""

# ══════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════
@app.post("/register")
async def register(req:RegReq):
    if len(req.password)<6: raise HTTPException(400,"Password min 6 characters.")
    conn = get_db()
    if conn.execute("SELECT id FROM users WHERE email=?",(req.email.lower().strip(),)).fetchone():
        conn.close(); raise HTTPException(400,"Email already registered.")
    uid = str(uuid.uuid4())
    conn.execute("INSERT INTO users (id,name,email,hashed_password,department) VALUES (?,?,?,?,?)",
        (uid,req.name.strip(),req.email.lower().strip(),pwd_ctx.hash(req.password),req.department))
    conn.commit(); conn.close()
    return {"success":True,"token":create_token(uid),"message":f"Welcome, {req.name}!"}

@app.post("/login")
async def login(req:LoginReq):
    conn = get_db()
    row  = conn.execute("SELECT * FROM users WHERE email=?",(req.email.lower().strip(),)).fetchone()
    conn.close()
    if not row: raise HTTPException(401,"No account with this email.")
    if not pwd_ctx.verify(req.password,row["hashed_password"]): raise HTTPException(401,"Incorrect password.")
    return {"success":True,"token":create_token(row["id"]),"message":f"Welcome back, {row['name']}!"}

@app.get("/me")
async def me(user=Depends(get_user)): return user

# ── Forgot Password ────────────────────────────────────────────────
@app.post("/forgot_password")
async def forgot_password(req:ForgotReq):
    conn = get_db()
    row  = conn.execute("SELECT id,name FROM users WHERE email=?",(req.email.lower().strip(),)).fetchone()
    if not row: conn.close(); raise HTTPException(404,"No account with this email.")
    otp     = str(secrets.randbelow(900000)+100000)  # 6 digit OTP
    expires = (datetime.utcnow()+timedelta(minutes=15)).isoformat()
    conn.execute("INSERT INTO password_resets (id,user_id,otp,expires_at) VALUES (?,?,?,?)",
        (str(uuid.uuid4()),row["id"],otp,expires))
    conn.commit(); conn.close()
    # In production send email — for demo return OTP
    log.info("OTP for %s: %s", req.email, otp)
    return {"success":True,"otp":otp,  # Remove in production!
            "message":f"OTP sent to {req.email}. Valid 15 minutes. (Demo: {otp})"}

@app.post("/verify_otp")
async def verify_otp(req:VerifyOTPReq):
    if len(req.new_password)<6: raise HTTPException(400,"Password min 6 characters.")
    conn = get_db()
    row  = conn.execute("SELECT pr.*,u.id user_id FROM password_resets pr JOIN users u ON pr.user_id=u.id WHERE u.email=? AND pr.otp=? AND pr.used=0",
        (req.email.lower().strip(),req.otp)).fetchone()
    if not row: conn.close(); raise HTTPException(400,"Invalid OTP.")
    if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
        conn.close(); raise HTTPException(400,"OTP expired. Request new one.")
    conn.execute("UPDATE users SET hashed_password=? WHERE id=?",(pwd_ctx.hash(req.new_password),row["user_id"]))
    conn.execute("UPDATE password_resets SET used=1 WHERE id=?",(row["id"],))
    conn.commit(); conn.close()
    return {"success":True,"message":"Password reset successfully! Please login."}

# ══════════════════════════════════════════════════════════════════
# CLASSROOMS
# ══════════════════════════════════════════════════════════════════
@app.post("/create_classroom")
async def create_classroom(req:CreateClsReq, user=Depends(get_user)):
    code = secrets.token_hex(3).upper()
    conn = get_db()
    while conn.execute("SELECT id FROM classrooms WHERE code=?",(code,)).fetchone():
        code = secrets.token_hex(3).upper()
    cid = str(uuid.uuid4())
    conn.execute("INSERT INTO classrooms (id,creator_id,name,subject,branch,year,section,code,description,banner_color) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (cid,user["id"],req.name,req.subject,req.branch,req.year,req.section,code,req.description,req.banner_color))
    conn.execute("INSERT INTO classroom_members (id,classroom_id,user_id,roll_number,is_admin,face_enrolled,face_locked) VALUES (?,?,?,?,?,?,?)",
        (str(uuid.uuid4()),cid,user["id"],"ADMIN",1,0,0))
    conn.commit(); conn.close()
    return {"success":True,"classroom_id":cid,"code":code,"message":f"Classroom '{req.name}' created! Code: {code}"}

@app.get("/classrooms")
async def get_classrooms(user=Depends(get_user)):
    conn = get_db()
    rows = conn.execute("""SELECT c.*,u.name creator_name,cm.is_admin,
        (SELECT COUNT(*) FROM classroom_members WHERE classroom_id=c.id) member_count,
        (SELECT COUNT(*) FROM assignments WHERE classroom_id=c.id AND due_date>=date('now')) upcoming_assignments
        FROM classrooms c JOIN users u ON c.creator_id=u.id
        JOIN classroom_members cm ON cm.classroom_id=c.id AND cm.user_id=?
        ORDER BY cm.joined_at DESC""",(user["id"],)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/classroom/{cid}")
async def get_classroom(cid:str, user=Depends(get_user)):
    conn = get_db()
    row  = conn.execute("SELECT c.*,u.name creator_name FROM classrooms c JOIN users u ON c.creator_id=u.id WHERE c.id=?",(cid,)).fetchone()
    if not row: conn.close(); raise HTTPException(404,"Classroom not found.")
    mem  = conn.execute("SELECT is_admin,face_enrolled,face_locked FROM classroom_members WHERE classroom_id=? AND user_id=?",(cid,user["id"])).fetchone()
    if not mem: conn.close(); raise HTTPException(403,"Not a member.")
    c = dict(row)
    c["is_admin"]     = bool(mem["is_admin"])
    c["face_enrolled"] = bool(mem["face_enrolled"])
    c["face_locked"]   = bool(mem["face_locked"])
    c["member_count"]  = conn.execute("SELECT COUNT(*) FROM classroom_members WHERE classroom_id=?",(cid,)).fetchone()[0]
    conn.close(); return c

@app.delete("/classroom/{cid}")
async def delete_classroom(cid:str, user=Depends(get_user)):
    require_admin(cid,user["id"])
    conn = get_db()
    for tbl in ["classroom_members","attendance","posts","comments","assignments","assignment_submissions","face_audit_logs"]:
        try: conn.execute(f"DELETE FROM {tbl} WHERE classroom_id=?",(cid,))
        except: pass
    conn.execute("DELETE FROM classrooms WHERE id=?",(cid,))
    conn.commit(); conn.close()
    store = load_faces()
    for k in [k for k in store if store[k].get("classroom_id")==cid]: del store[k]
    save_faces(store)
    return {"success":True,"message":"Classroom deleted."}

@app.get("/classroom/{cid}/members")
async def get_members(cid:str, user=Depends(get_user)):
    conn = get_db()
    rows = conn.execute("""SELECT u.id,u.name,u.email,cm.roll_number,cm.branch,cm.year,cm.section,
        cm.face_enrolled,cm.face_locked,cm.is_admin,cm.joined_at,cm.face_updated_by,cm.face_updated_at
        FROM classroom_members cm JOIN users u ON cm.user_id=u.id
        WHERE cm.classroom_id=? ORDER BY cm.is_admin DESC,u.name""",(cid,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.delete("/classroom/{cid}/remove/{uid}")
async def remove_member(cid:str, uid:str, user=Depends(get_user)):
    require_admin(cid,user["id"])
    if uid==user["id"]: raise HTTPException(400,"Cannot remove yourself.")
    conn = get_db()
    conn.execute("DELETE FROM classroom_members WHERE classroom_id=? AND user_id=?",(cid,uid))
    conn.commit(); conn.close()
    store = load_faces()
    store.pop(f"cls_{cid}_stu_{uid}",None)
    save_faces(store)
    return {"success":True,"message":"Member removed."}

# ══════════════════════════════════════════════════════════════════
# JOIN + FACE ENROLLMENT (with face lock)
# ══════════════════════════════════════════════════════════════════
@app.post("/join_classroom")
async def join_classroom(req:JoinClsReq, user=Depends(get_user)):
    conn = get_db()
    cls  = conn.execute("SELECT * FROM classrooms WHERE code=?",(req.code.strip().upper(),)).fetchone()
    if not cls: conn.close(); raise HTTPException(404,"Invalid classroom code.")
    cid  = cls["id"]
    if conn.execute("SELECT id FROM classroom_members WHERE classroom_id=? AND user_id=?",(cid,user["id"])).fetchone():
        conn.close(); raise HTTPException(400,"Already a member.")
    mid = str(uuid.uuid4())
    conn.execute("INSERT INTO classroom_members (id,classroom_id,user_id,roll_number,branch,year,section,is_admin) VALUES (?,?,?,?,?,?,?,0)",
        (mid,cid,user["id"],req.roll_number,req.branch,req.year,req.section))
    conn.commit()
    face_enrolled=0; face_msg=""
    if req.image:
        if fr is None:
            face_msg="⚠️ face_recognition not installed."
        else:
            try:
                rgb = b64_to_rgb(req.image)
                enc = encode_face(rgb)
                store = load_faces()
                store[f"cls_{cid}_stu_{user['id']}"] = {
                    "encoding":enc,"student_id":user["id"],"classroom_id":cid,
                    "name":user["name"],"roll":req.roll_number}
                save_faces(store)
                now = datetime.now().isoformat()
                conn.execute("UPDATE classroom_members SET face_enrolled=1,face_locked=1,face_updated_by=?,face_updated_at=? WHERE id=?",
                    (user["id"],now,mid))
                conn.commit()
                # Audit log
                conn.execute("INSERT INTO face_audit_logs (id,classroom_id,student_id,action,performed_by,notes) VALUES (?,?,?,?,?,?)",
                    (str(uuid.uuid4()),cid,user["id"],"ENROLLED",user["id"],"Self-enrollment during join"))
                conn.commit()
                face_enrolled=1; face_msg="✅ Face enrolled and locked!"
                log.info("✅ Face enrolled+locked: %s in %s",user["name"],cid)
            except ValueError as e:
                face_msg=f"⚠️ {e}"
            except Exception as e:
                face_msg=f"⚠️ Error: {e}"
    else:
        face_msg="No face captured. Enroll face for attendance."
    conn.close()
    return {"success":True,"classroom_id":cid,"face_enrolled":face_enrolled,
            "message":f"Joined '{cls['name']}'! {face_msg}"}

@app.post("/enroll_face")
async def enroll_face(classroom_id:str=Form(...), image:str=Form(...), user=Depends(get_user)):
    """Student self-enroll — only if not already locked."""
    if fr is None: raise HTTPException(500,"face_recognition not installed.")
    conn = get_db()
    row  = conn.execute("SELECT * FROM classroom_members WHERE classroom_id=? AND user_id=?",(classroom_id,user["id"])).fetchone()
    if not row: conn.close(); raise HTTPException(404,"Not a member.")
    # SECURITY: If already enrolled and locked, block
    if row["face_enrolled"] and row["face_locked"]:
        conn.close()
        raise HTTPException(403,"Face already registered and locked. Contact your classroom admin to reset.")
    try:
        enc = encode_face(b64_to_rgb(image))
    except ValueError as e:
        conn.close(); raise HTTPException(400,str(e))
    store = load_faces()
    store[f"cls_{classroom_id}_stu_{user['id']}"] = {
        "encoding":enc,"student_id":user["id"],"classroom_id":classroom_id,
        "name":user["name"],"roll":row["roll_number"]}
    save_faces(store)
    now = datetime.now().isoformat()
    conn.execute("UPDATE classroom_members SET face_enrolled=1,face_locked=1,face_updated_by=?,face_updated_at=? WHERE classroom_id=? AND user_id=?",
        (user["id"],now,classroom_id,user["id"]))
    conn.execute("INSERT INTO face_audit_logs (id,classroom_id,student_id,action,performed_by,notes) VALUES (?,?,?,?,?,?)",
        (str(uuid.uuid4()),classroom_id,user["id"],"ENROLLED",user["id"],"Student self-enrollment"))
    conn.commit(); conn.close()
    return {"success":True,"message":"Face enrolled and locked! Admin must reset if you need to change it."}

@app.post("/admin_reset_face")
async def admin_reset_face(req:FaceResetReq, user=Depends(get_user)):
    """Admin-only: reset and re-enroll a student's face."""
    require_admin(req.classroom_id,user["id"])
    if fr is None: raise HTTPException(500,"face_recognition not installed.")
    conn = get_db()
    row  = conn.execute("SELECT * FROM classroom_members WHERE classroom_id=? AND user_id=?",(req.classroom_id,req.student_id)).fetchone()
    if not row: conn.close(); raise HTTPException(404,"Student not found in classroom.")
    try:
        enc = encode_face(b64_to_rgb(req.image))
    except ValueError as e:
        conn.close(); raise HTTPException(400,str(e))
    store = load_faces()
    store[f"cls_{req.classroom_id}_stu_{req.student_id}"] = {
        "encoding":enc,"student_id":req.student_id,"classroom_id":req.classroom_id,
        "name":row["name"],"roll":row["roll_number"]}
    save_faces(store)
    now = datetime.now().isoformat()
    conn.execute("UPDATE classroom_members SET face_enrolled=1,face_locked=1,face_updated_by=?,face_updated_at=? WHERE classroom_id=? AND user_id=?",
        (user["id"],now,req.classroom_id,req.student_id))
    conn.execute("INSERT INTO face_audit_logs (id,classroom_id,student_id,action,performed_by,notes) VALUES (?,?,?,?,?,?)",
        (str(uuid.uuid4()),req.classroom_id,req.student_id,"ADMIN_RESET",user["id"],req.notes or "Admin reset"))
    conn.commit()
    # Notify student
    send_notification(req.student_id,"Face Data Reset",
        "Your face registration has been reset by your classroom admin. Please re-enroll if needed.","warn")
    conn.close()
    log.info("✅ Admin face reset: student=%s by admin=%s",req.student_id,user["id"])
    return {"success":True,"message":"Face data reset and updated successfully!"}

@app.post("/admin_clear_face")
async def admin_clear_face(classroom_id:str=Form(...), student_id:str=Form(...),
                           notes:str=Form(""), user=Depends(get_user)):
    """Admin: clear face data so student can re-enroll themselves."""
    require_admin(classroom_id,user["id"])
    conn = get_db()
    conn.execute("UPDATE classroom_members SET face_enrolled=0,face_locked=0,face_updated_by=?,face_updated_at=? WHERE classroom_id=? AND user_id=?",
        (user["id"],datetime.now().isoformat(),classroom_id,student_id))
    conn.execute("INSERT INTO face_audit_logs (id,classroom_id,student_id,action,performed_by,notes) VALUES (?,?,?,?,?,?)",
        (str(uuid.uuid4()),classroom_id,student_id,"ADMIN_CLEARED",user["id"],notes or "Admin cleared face"))
    conn.commit()
    store = load_faces()
    store.pop(f"cls_{classroom_id}_stu_{student_id}",None)
    save_faces(store)
    send_notification(student_id,"Face Data Cleared","Your face data was cleared by admin. You can now re-enroll your face.","info")
    conn.close()
    return {"success":True,"message":"Face data cleared. Student can re-enroll."}

# Purane @app.get("/face_audit_logs/{classroom_id}") ko isse badal dein:
@app.get("/admin/face_audit/{classroom_id}")
async def face_audit_logs(classroom_id:str, user=Depends(get_uid)): # get_uid use karein authentication ke liye
    require_admin(classroom_id, user)
    conn = get_db()
    rows = conn.execute("""SELECT f.*, u1.name student_name, u2.name performed_by_name
        FROM face_audit_logs f
        LEFT JOIN users u1 ON f.student_id=u1.id
        LEFT JOIN users u2 ON f.performed_by=u2.id
        WHERE f.classroom_id=? ORDER BY f.performed_at DESC""", (classroom_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
# ══════════════════════════════════════════════════════════════════
# ATTENDANCE
# ══════════════════════════════════════════════════════════════════
@app.post("/recognize")
async def recognize(req:RecognizeReq, user=Depends(get_user)):
    require_admin(req.classroom_id,user["id"])
    if fr is None: raise HTTPException(500,"face_recognition not installed.")
    store   = load_faces()
    prefix  = f"cls_{req.classroom_id}_stu_"
    members = {k:v for k,v in store.items() if k.startswith(prefix)}
    log.info("Classroom members in pkl: %d", len(members))
    if not members:
        return {"results":[{"status":"error","message":"No students with face data. Ask students to enroll face."}]}
    try: rgb = b64_to_rgb(req.image)
    except Exception as e: raise HTTPException(400,f"Invalid image: {e}")
    locs = fr.face_locations(rgb,model="hog")
    if not locs: return {"status":"no_face","results":[]}
    frame_encs = fr.face_encodings(rgb,known_face_locations=locs,num_jitters=1)
    keys       = list(members.keys())
    known_encs = [members[k]["encoding"] for k in keys]
    results    = []; today=date.today().isoformat(); now=datetime.now().strftime("%H:%M:%S")
    for fenc in frame_encs:
        matches   = fr.compare_faces(known_encs,fenc,tolerance=TOLERANCE)
        distances = fr.face_distance(known_encs,fenc)
        if not any(matches):
            results.append({"status":"unknown","message":"Unknown student."}); continue
        idx  = int(np.argmin(distances)); dist=float(distances[idx])
        conf = round((1.0-dist)*100,1); meta=members[keys[idx]]
        conn = get_db()
        try:
            conn.execute("INSERT INTO attendance (id,classroom_id,student_id,date,time,status,confidence) VALUES (?,?,?,?,?,?,?)",
                (str(uuid.uuid4()),req.classroom_id,meta["student_id"],today,now,"present",conf))
            conn.commit()
            results.append({"status":"present","name":meta["name"],"roll":meta["roll"],"confidence":conf,"date":today,"time":now})
            log.info("✅ Attendance: %s %.1f%%",meta["name"],conf)
        except sqlite3.IntegrityError:
            results.append({"status":"duplicate","name":meta["name"],"roll":meta["roll"],"confidence":conf,"message":f"{meta['name']} already marked."})
        except Exception as e: results.append({"status":"error","message":str(e)})
        finally: conn.close()
    return {"results":results}

@app.get("/attendance/{classroom_id}")
async def get_attendance(classroom_id:str, date_filter:Optional[str]=None, user=Depends(get_user)):
    conn = get_db()
    mem  = conn.execute("SELECT is_admin FROM classroom_members WHERE classroom_id=? AND user_id=?",(classroom_id,user["id"])).fetchone()
    if not mem: conn.close(); raise HTTPException(403,"Not a member.")
    if mem["is_admin"]:
        q="SELECT a.*,u.name student_name FROM attendance a JOIN users u ON a.student_id=u.id WHERE a.classroom_id=?"; params=[classroom_id]
    else:
        q="SELECT a.*,u.name student_name FROM attendance a JOIN users u ON a.student_id=u.id WHERE a.classroom_id=? AND a.student_id=?"; params=[classroom_id,user["id"]]
    if date_filter: q+=" AND a.date=?"; params.append(date_filter)
    q+=" ORDER BY a.date DESC,a.time DESC"
    rows=conn.execute(q,params).fetchall(); conn.close()
    return [dict(r) for r in rows]

@app.get("/my_attendance")
async def my_attendance(user=Depends(get_user)):
    conn = get_db()
    rows = conn.execute("""SELECT a.*,c.name classroom_name,c.subject FROM attendance a
        JOIN classrooms c ON a.classroom_id=c.id WHERE a.student_id=? ORDER BY a.date DESC""",(user["id"],)).fetchall()
    conn.close(); return [dict(r) for r in rows]

@app.post("/mark_absent/{classroom_id}")
async def mark_absent(classroom_id:str, user=Depends(get_user)):
    require_admin(classroom_id,user["id"])
    conn=get_db(); today=date.today().isoformat(); now=datetime.now().strftime("%H:%M:%S")
    mems=conn.execute("SELECT user_id FROM classroom_members WHERE classroom_id=? AND is_admin=0",(classroom_id,)).fetchall()
    count=0
    for m in mems:
        sid=m["user_id"]
        if not conn.execute("SELECT id FROM attendance WHERE classroom_id=? AND student_id=? AND date=?",(classroom_id,sid,today)).fetchone():
            conn.execute("INSERT INTO attendance (id,classroom_id,student_id,date,time,status,confidence) VALUES (?,?,?,?,?,?,?)",
                (str(uuid.uuid4()),classroom_id,sid,today,now,"absent",0)); count+=1
    conn.commit(); conn.close()
    return {"success":True,"message":f"{count} students marked absent."}

# ══════════════════════════════════════════════════════════════════
# ASSIGNMENTS — including across all classrooms
# ══════════════════════════════════════════════════════════════════
@app.get("/my_assignments")
async def my_assignments(user=Depends(get_user)):
    """Get all assignments across all joined classrooms."""
    conn  = get_db(); today=date.today().isoformat()
    cls_ids = [r["classroom_id"] for r in conn.execute(
        "SELECT classroom_id FROM classroom_members WHERE user_id=?",(user["id"],)).fetchall()]
    result = []
    for cid in cls_ids:
        cls_row = conn.execute("SELECT name,subject FROM classrooms WHERE id=?",(cid,)).fetchone()
        if not cls_row: continue
        mem = conn.execute("SELECT is_admin FROM classroom_members WHERE classroom_id=? AND user_id=?",(cid,user["id"])).fetchone()
        is_admin = mem and mem["is_admin"]
        rows = conn.execute("SELECT * FROM assignments WHERE classroom_id=? ORDER BY due_date",(cid,)).fetchall()
        for r in rows:
            a = dict(r)
            a["classroom_name"] = cls_row["name"]
            a["subject"]        = cls_row["subject"]
            a["is_overdue"]     = a["due_date"] < today
            a["is_admin"]       = bool(is_admin)
            if not is_admin:
                sub = conn.execute("SELECT * FROM assignment_submissions WHERE assignment_id=? AND student_id=?",(a["id"],user["id"])).fetchone()
                a["my_submission"] = dict(sub) if sub else None
            else:
                a["submission_count"] = conn.execute("SELECT COUNT(*) FROM assignment_submissions WHERE assignment_id=?",(a["id"],)).fetchone()[0]
            result.append(a)
    conn.close()
    result.sort(key=lambda x: x["due_date"])
    return result

@app.post("/create_assignment")
async def create_assignment(classroom_id:str=Form(...),title:str=Form(...),
    description:str=Form(""),due_date:str=Form(...),
    file:UploadFile=File(None), user=Depends(get_user)):
    require_admin(classroom_id,user["id"])
    file_url=file_name=""
    if file and file.filename:
        ext=os.path.splitext(file.filename)[1]; fn=f"{uuid.uuid4().hex}{ext}"
        with open(os.path.join(UPLOAD_DIR,fn),"wb") as fw: fw.write(await file.read())
        file_url=f"/uploads/{fn}"; file_name=file.filename
    aid=str(uuid.uuid4()); conn=get_db()
    conn.execute("INSERT INTO assignments (id,classroom_id,creator_id,title,description,file_url,file_name,due_date) VALUES (?,?,?,?,?,?,?,?)",
        (aid,classroom_id,user["id"],title,description,file_url,file_name,due_date))
    conn.commit()
    # Notify all members
    mems=conn.execute("SELECT user_id FROM classroom_members WHERE classroom_id=? AND is_admin=0",(classroom_id,)).fetchall()
    cls=conn.execute("SELECT name FROM classrooms WHERE id=?",(classroom_id,)).fetchone()
    for m in mems:
        send_notification(m["user_id"],"New Assignment",
            f"New assignment '{title}' in {cls['name']}. Due: {due_date}","info")
    conn.close()
    return {"success":True,"assignment_id":aid,"message":"Assignment created!"}

@app.get("/assignments/{classroom_id}")
async def get_assignments(classroom_id:str, user=Depends(get_user)):
    conn=get_db(); today=date.today().isoformat()
    mem=conn.execute("SELECT is_admin FROM classroom_members WHERE classroom_id=? AND user_id=?",(classroom_id,user["id"])).fetchone()
    is_admin=mem and mem["is_admin"]
    rows=conn.execute("SELECT * FROM assignments WHERE classroom_id=? ORDER BY due_date",(classroom_id,)).fetchall()
    result=[]
    for r in rows:
        a=dict(r); a["is_overdue"]=a["due_date"]<today
        if is_admin: a["submission_count"]=conn.execute("SELECT COUNT(*) FROM assignment_submissions WHERE assignment_id=?",(a["id"],)).fetchone()[0]
        else:
            sub=conn.execute("SELECT * FROM assignment_submissions WHERE assignment_id=? AND student_id=?",(a["id"],user["id"])).fetchone()
            a["my_submission"]=dict(sub) if sub else None
        result.append(a)
    conn.close(); return result

@app.put("/assignment/{aid}/due_date")
async def update_due(aid:str, req:UpdateDueDateReq, user=Depends(get_user)):
    conn=get_db(); a=conn.execute("SELECT classroom_id FROM assignments WHERE id=?",(aid,)).fetchone()
    if not a: conn.close(); raise HTTPException(404,"Not found.")
    require_admin(a["classroom_id"],user["id"])
    conn.execute("UPDATE assignments SET due_date=? WHERE id=?",(req.due_date,aid))
    conn.commit(); conn.close()
    return {"success":True,"message":"Due date updated."}

@app.delete("/assignment/{aid}")
async def delete_assignment(aid:str, user=Depends(get_user)):
    conn=get_db(); a=conn.execute("SELECT classroom_id FROM assignments WHERE id=?",(aid,)).fetchone()
    if not a: conn.close(); raise HTTPException(404,"Not found.")
    require_admin(a["classroom_id"],user["id"])
    conn.execute("DELETE FROM assignment_submissions WHERE assignment_id=?",(aid,))
    conn.execute("DELETE FROM assignments WHERE id=?",(aid,))
    conn.commit(); conn.close()
    return {"success":True}

@app.post("/submit_assignment")
async def submit_assignment(assignment_id:str=Form(...),file:UploadFile=File(None),user=Depends(get_user)):
    conn=get_db(); asgn=conn.execute("SELECT * FROM assignments WHERE id=?",(assignment_id,)).fetchone()
    if not asgn: conn.close(); raise HTTPException(404,"Assignment not found.")
    today=date.today().isoformat(); sv="submitted" if asgn["due_date"]>=today else "late"
    file_url=file_name=""
    if file and file.filename:
        ext=os.path.splitext(file.filename)[1]; fn=f"{uuid.uuid4().hex}{ext}"
        with open(os.path.join(UPLOAD_DIR,fn),"wb") as fw: fw.write(await file.read())
        file_url=f"/uploads/{fn}"; file_name=file.filename
    now=datetime.now().isoformat()
    try:
        conn.execute("INSERT INTO assignment_submissions (id,assignment_id,student_id,file_url,file_name,submitted_at,status) VALUES (?,?,?,?,?,?,?)",
            (str(uuid.uuid4()),assignment_id,user["id"],file_url,file_name,now,sv))
    except sqlite3.IntegrityError:
        conn.execute("UPDATE assignment_submissions SET file_url=?,file_name=?,submitted_at=?,status=? WHERE assignment_id=? AND student_id=?",
            (file_url,file_name,now,sv,assignment_id,user["id"]))
    conn.commit(); conn.close()
    return {"success":True,"status":sv,"message":f"Submitted {'on time' if sv=='submitted' else 'late'}!"}

@app.get("/submissions/{assignment_id}")
async def get_submissions(assignment_id:str, user=Depends(get_user)):
    conn=get_db(); a=conn.execute("SELECT classroom_id FROM assignments WHERE id=?",(assignment_id,)).fetchone()
    if not a: conn.close(); raise HTTPException(404,"Not found.")
    require_admin(a["classroom_id"],user["id"])
    rows=conn.execute("SELECT s.*,u.name student_name FROM assignment_submissions s JOIN users u ON s.student_id=u.id WHERE s.assignment_id=? ORDER BY s.submitted_at",(assignment_id,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

# ══════════════════════════════════════════════════════════════════
# POSTS / COMMENTS
# ══════════════════════════════════════════════════════════════════
@app.post("/post")
async def create_post(req:PostReq, user=Depends(get_user)):
    require_admin(req.classroom_id,user["id"])
    pid=str(uuid.uuid4()); conn=get_db()
    conn.execute("INSERT INTO posts (id,classroom_id,user_id,type,title,content) VALUES (?,?,?,?,?,?)",
        (pid,req.classroom_id,user["id"],req.type,req.title,req.content))
    conn.commit(); conn.close()
    return {"success":True,"post_id":pid}

@app.post("/upload_material")
async def upload_material(classroom_id:str=Form(...),title:str=Form(...),
    content:str=Form(""),file:UploadFile=File(None),user=Depends(get_user)):
    require_admin(classroom_id,user["id"])
    file_url=file_name=""
    if file and file.filename:
        ext=os.path.splitext(file.filename)[1]; fn=f"{uuid.uuid4().hex}{ext}"
        with open(os.path.join(UPLOAD_DIR,fn),"wb") as fw: fw.write(await file.read())
        file_url=f"/uploads/{fn}"; file_name=file.filename
    pid=str(uuid.uuid4()); conn=get_db()
    conn.execute("INSERT INTO posts (id,classroom_id,user_id,type,title,content,file_url,file_name) VALUES (?,?,?,?,?,?,?,?)",
        (pid,classroom_id,user["id"],"material",title,content,file_url,file_name))
    conn.commit(); conn.close()
    return {"success":True,"post_id":pid,"file_url":file_url,"message":"Uploaded!"}

@app.get("/posts/{classroom_id}")
async def get_posts(classroom_id:str, user=Depends(get_user)):
    conn=get_db()
    rows=conn.execute("""SELECT p.*,u.name user_name,(SELECT COUNT(*) FROM comments WHERE post_id=p.id) comment_count
        FROM posts p JOIN users u ON p.user_id=u.id WHERE p.classroom_id=? ORDER BY p.created_at DESC""",(classroom_id,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

@app.delete("/post/{pid}")
async def delete_post(pid:str, user=Depends(get_user)):
    conn=get_db(); post=conn.execute("SELECT classroom_id FROM posts WHERE id=?",(pid,)).fetchone()
    if not post: conn.close(); raise HTTPException(404,"Post not found.")
    require_admin(post["classroom_id"],user["id"])
    conn.execute("DELETE FROM comments WHERE post_id=?",(pid,))
    conn.execute("DELETE FROM posts WHERE id=?",(pid,))
    conn.commit(); conn.close(); return {"success":True}

@app.post("/comment")
async def add_comment(req:CommentReq, user=Depends(get_user)):
    conn=get_db()
    conn.execute("INSERT INTO comments (id,post_id,user_id,comment) VALUES (?,?,?,?)",
        (str(uuid.uuid4()),req.post_id,user["id"],req.comment))
    conn.commit(); conn.close(); return {"success":True}

@app.get("/comments/{post_id}")
async def get_comments(post_id:str, user=Depends(get_user)):
    conn=get_db()
    rows=conn.execute("SELECT c.*,u.name user_name FROM comments c JOIN users u ON c.user_id=u.id WHERE c.post_id=? ORDER BY c.created_at",(post_id,)).fetchall()
    conn.close(); return [dict(r) for r in rows]

# ══════════════════════════════════════════════════════════════════
# NOTIFICATIONS
# ══════════════════════════════════════════════════════════════════
@app.get("/notifications")
async def get_notifications(user=Depends(get_user)):
    conn=get_db()
    rows=conn.execute("SELECT * FROM notifications WHERE user_id=? ORDER BY created_at DESC LIMIT 50",(user["id"],)).fetchall()
    conn.close(); return [dict(r) for r in rows]

@app.post("/notifications/read_all")
async def read_all_notifications(user=Depends(get_user)):
    conn=get_db()
    conn.execute("UPDATE notifications SET read=1 WHERE user_id=?",(user["id"],))
    conn.commit(); conn.close()
    return {"success":True}

# ══════════════════════════════════════════════════════════════════
# STATS & HEALTH
# ══════════════════════════════════════════════════════════════════
@app.get("/stats/{classroom_id}")
async def stats(classroom_id:str, user=Depends(get_user)):
    conn=get_db(); today=date.today().isoformat()
    store=load_faces()
    d={
        "members":       conn.execute("SELECT COUNT(*) FROM classroom_members WHERE classroom_id=?",(classroom_id,)).fetchone()[0],
        "face_enrolled": conn.execute("SELECT COUNT(*) FROM classroom_members WHERE classroom_id=? AND face_enrolled=1",(classroom_id,)).fetchone()[0],
        "face_locked":   conn.execute("SELECT COUNT(*) FROM classroom_members WHERE classroom_id=? AND face_locked=1",(classroom_id,)).fetchone()[0],
        "face_in_pkl":   sum(1 for k in store if k.startswith(f"cls_{classroom_id}_stu_")),
        "today_present": conn.execute("SELECT COUNT(*) FROM attendance WHERE classroom_id=? AND date=? AND status='present'",(classroom_id,today)).fetchone()[0],
        "total_posts":   conn.execute("SELECT COUNT(*) FROM posts WHERE classroom_id=?",(classroom_id,)).fetchone()[0],
        "total_assignments":conn.execute("SELECT COUNT(*) FROM assignments WHERE classroom_id=?",(classroom_id,)).fetchone()[0],
        "total_records": conn.execute("SELECT COUNT(*) FROM attendance WHERE classroom_id=?",(classroom_id,)).fetchone()[0],
    }
    conn.close(); return d

@app.get("/health")
async def health():
    store=load_faces()
    return {"status":"ok","version":"9.0.0","face_recognition":"installed" if fr else "NOT INSTALLED",
            "total_encodings":len(store),"timestamp":datetime.now().isoformat()}









from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Agar aapki photo 'assets' folder mein hai
@app.get("/get-logo")
async def get_logo():
    return FileResponse(r'C:\Users\VIKASH YADAV\Desktop\all nitj\nitja1\download.jpg')



from email_validator import validate_email, EmailNotValidError

@app.post("/register")
async def register(req: RegReq):
    try:
        # Check if email is valid and has deliverable MX records
        email_info = validate_email(req.email, check_deliverability=True)
        email = email_info.normalized
    except EmailNotValidError as e:
        raise HTTPException(400, f"Invalid or non-existent email: {str(e)}")
    
    # ... Proceed to check DB and save