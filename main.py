import os
import re
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Header
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship

from passlib.context import CryptContext
from dotenv import load_dotenv
import pypdf
import io

from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Environment & Configuration
# ---------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# Create local uploads folder if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ai_client = None
if GEMINI_API_KEY:
    ai_client = genai.Client(api_key=GEMINI_API_KEY)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB Limit

# ---------------------------------------------------------------------------
# Helper Function: Text Sanitization (Prevents LaTeX Render Bugs)
# ---------------------------------------------------------------------------


def sanitize_medical_text(text: str) -> str:
    """
    Post-processing safeguard: Strips LaTeX math syntax and stray dollar signs 
    from LLM outputs to prevent rendering bugs on the React frontend.
    """
    if not text:
        return text

    # Remove LaTeX \text{...} wrappers (e.g., \text{mg} -> mg, \text{mL} -> mL)
    text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)

    # Remove any remaining LaTeX backslashes before units
    text = re.sub(r'\\([a-zA-Z]+)', r'\1', text)

    # Strip inline math dollar signs surrounding numbers/units (e.g., $5ml$ -> 5ml)
    text = re.sub(r'\$([^$]+)\$', r'\1', text)

    # Clean up isolated dollar signs
    text = text.replace('$', '')

    return text


# ---------------------------------------------------------------------------
# 2. Database Setup & ORM Models
# ---------------------------------------------------------------------------
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), default="patient")
    created_at = Column(DateTime, default=datetime.utcnow)

    appointments = relationship("Appointment", back_populates="patient")
    medical_records = relationship("MedicalRecord", back_populates="patient")


class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    doctor_name = Column(String(100), nullable=False)
    appointment_date = Column(String(50), nullable=False)
    notes = Column(Text, nullable=True)

    patient = relationship("User", back_populates="appointments")


class MedicalRecord(Base):
    __tablename__ = "medical_records"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    # Stores path to original file
    file_path = Column(String(500), nullable=True)
    # Stores original filename
    file_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("User", back_populates="medical_records")


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 3. Security / Hashing Setup
# ---------------------------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# ---------------------------------------------------------------------------
# 4. Pydantic Schemas
# ---------------------------------------------------------------------------


class UserCreate(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    role: Optional[str] = "patient"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class AppointmentCreate(BaseModel):
    doctor_name: str
    appointment_date: str
    reason: Optional[str] = None


class TriageRequest(BaseModel):
    symptoms: str


class RecommendationRequest(BaseModel):
    concern_or_goal: str


# ---------------------------------------------------------------------------
# 5. FastAPI App Setup
# ---------------------------------------------------------------------------
app = FastAPI(title="MediCore AI Clinical Portal")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return RedirectResponse(url="/static/login.html")


app.mount("/static", StaticFiles(directory="static"), name="static")

# ---------------------------------------------------------------------------
# 6. API Endpoints
# ---------------------------------------------------------------------------

# --- System & Keep-Alive Monitoring ---


@app.api_route("/health", methods=["GET", "HEAD"])
@app.api_route("/health/", methods=["GET", "HEAD"])
def health_check():
    """
    Lightweight health check endpoint.
    Executes 'SELECT 1' to keep Aiven MySQL active 
    and returns 200 OK to keep Render web service awake.
    Supports both GET and HEAD methods for UptimeRobot.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# --- Authentication ---


@app.post("/api/register")
@app.post("/api/register/")
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(
        User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed_pwd = hash_password(user_data.password)
    new_user = User(
        full_name=user_data.full_name,
        email=user_data.email,
        hashed_password=hashed_pwd,
        role=user_data.role
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/api/login")
@app.post("/api/login/")
def login_user(login_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401, detail="Invalid email or password")

    return {
        "message": "Login successful",
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "email": user.email,
            "role": user.role
        }
    }

# --- Clinical AI Features ---


@app.post("/api/triage")
@app.post("/api/triage/")
@app.post("/api/ai/triage")
@app.post("/api/ai/triage/")
def ai_symptom_triage(request: TriageRequest):
    if not ai_client:
        raise HTTPException(
            status_code=500, detail="Gemini API Key is not configured.")

    prompt = (
        f"You are an AI clinical triage assistant for MediCore AI portal. "
        f"Analyze the following patient symptoms. Format your answer nicely using Markdown headings, "
        f"bullet points, and clear sections (Assessment, Urgency Level, Next Steps).\n"
        f"CRITICAL: Do NOT use LaTeX or dollar signs ($) for any units or numbers (e.g. write 5ml, 37.8 C, 500mg).\n\n"
        f"Symptoms: {request.symptoms}"
    )

    try:
        response = ai_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        clean_text = sanitize_medical_text(response.text)
        return {"assessment": clean_text, "triage_assessment": clean_text}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"AI Service Error: {str(e)}")


@app.post("/api/recommendations")
@app.post("/api/recommendations/")
def ai_health_recommendations(request: RecommendationRequest):
    if not ai_client:
        raise HTTPException(
            status_code=500, detail="Gemini API Key is not configured.")

    prompt = (
        f"You are an expert health and wellness advisor for MediCore AI. "
        f"Provide structured health recommendations using Markdown formatting with bullet points "
        f"and concise categories. Do NOT use LaTeX syntax or dollar signs ($) for measurements.\n\n"
        f"Concern/Goal: {request.concern_or_goal}"
    )

    try:
        response = ai_client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        clean_text = sanitize_medical_text(response.text)
        return {"recommendations": clean_text}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"AI Service Error: {str(e)}")


@app.post("/api/upload-report")
@app.post("/api/upload-report/")
async def upload_report(
    file: UploadFile = File(...),
    x_user_email: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400, detail="File size exceeds the 10 MB limit.")

    filename = file.filename.lower()
    allowed_exts = (".pdf", ".jpg", ".jpeg", ".png")
    if not any(filename.endswith(ext) for ext in allowed_exts):
        raise HTTPException(
            status_code=400, detail="Only PDF, JPG, JPEG, and PNG files are allowed.")

    # 1. Save original file to disk
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_filename = f"{timestamp}_{file.filename}"
    saved_file_path = os.path.join(UPLOAD_DIR, saved_filename)

    with open(saved_file_path, "wb") as f:
        f.write(contents)

    # 2. Extract AI summary
    summary = ""
    try:
        if filename.endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(contents))
            extracted_text = "".join(
                [page.extract_text() or "" for page in reader.pages])
            if not extracted_text.strip():
                extracted_text = "Medical Document PDF"

            if ai_client:
                prompt = (
                    f"Summarize this medical report in clean Markdown format with key observations and metrics.\n"
                    f"CRITICAL RULES:\n"
                    f"1. Strictly DO NOT use LaTeX or dollar sign ($) formatting for units.\n"
                    f"2. Write all dosage/units in simple plain text (e.g. '5ml', '500mg/5ml', '37.8 C').\n\n"
                    f"Document Text:\n{extracted_text}"
                )
                resp = ai_client.models.generate_content(
                    model="gemini-3.6-flash", contents=prompt)
                summary = sanitize_medical_text(resp.text)
            else:
                summary = "PDF parsed successfully (AI client unconfigured)."

        else:  # Image (JPG, JPEG, PNG)
            if ai_client:
                mime_type = file.content_type or "image/jpeg"
                image_part = types.Part.from_bytes(
                    data=contents, mime_type=mime_type)

                prompt = (
                    "Examine this handwritten or typed medical prescription/report image carefully.\n"
                    "Summarize key findings, patient details, diagnoses, and medication dosages in structured Markdown.\n"
                    "CRITICAL RULES:\n"
                    "1. Strictly DO NOT use LaTeX formatting, backslashes, or dollar signs ($) for any unit or dosage.\n"
                    "2. Express dosages, quantities, and units in plain text only (e.g. write '5ml', '500mg', '5ml - 5ml - 5ml').\n"
                    "3. Pay close attention to handwritten dosage numbers."
                )

                resp = ai_client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=[image_part, prompt]
                )
                summary = sanitize_medical_text(resp.text)
            else:
                summary = "Image uploaded successfully (AI client unconfigured)."

        # 3. Save to DB with file metadata
        patient = None
        record_id = None
        if x_user_email:
            patient = db.query(User).filter(User.email == x_user_email).first()

        if patient:
            new_record = MedicalRecord(
                patient_id=patient.id,
                title=f"Report Analysis ({file.filename})",
                description=summary,
                file_path=saved_file_path,
                file_name=file.filename
            )
            db.add(new_record)
            db.commit()
            db.refresh(new_record)
            record_id = new_record.id

        return {
            "filename": file.filename,
            "summary": summary,
            "record_id": record_id,
            "has_file": True
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"File processing error: {str(e)}")

# --- Appointments ---


@app.post("/api/appointments")
@app.post("/api/appointments/")
def create_appointment(
    appointment: AppointmentCreate,
    x_user_email: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    patient = None
    if x_user_email:
        patient = db.query(User).filter(User.email == x_user_email).first()

    if not patient:
        raise HTTPException(
            status_code=404, detail="Authenticated user not found.")

    new_app = Appointment(
        patient_id=patient.id,
        doctor_name=appointment.doctor_name,
        appointment_date=appointment.appointment_date,
        notes=appointment.reason or "Routine Consult"
    )
    db.add(new_app)
    db.commit()
    db.refresh(new_app)

    # Log into EHR history
    ehr_rec = MedicalRecord(
        patient_id=patient.id,
        title=f"Appointment with Dr. {appointment.doctor_name}",
        description=f"Date: {appointment.appointment_date} | Reason: {appointment.reason or 'Consultation'}"
    )
    db.add(ehr_rec)
    db.commit()

    return {"message": "Appointment scheduled successfully", "appointment_id": new_app.id}

# --- EHR History & File Downloads ---


@app.get("/api/ehr/{user_email}")
@app.get("/api/ehr/{user_email}/")
@app.get("/api/ehr")
@app.get("/api/ehr/")
def get_ehr_history(user_email: Optional[str] = None, db: Session = Depends(get_db)):
    if not user_email or user_email in ["null", "undefined"]:
        return []

    requesting_user = db.query(User).filter(User.email == user_email).first()
    if not requesting_user:
        return []

    if requesting_user.role in ["doctor", "admin"]:
        records = db.query(MedicalRecord).order_by(
            MedicalRecord.created_at.desc()).all()
    else:
        records = db.query(MedicalRecord).filter(
            MedicalRecord.patient_id == requesting_user.id
        ).order_by(MedicalRecord.created_at.desc()).all()

    formatted_records = [
        {
            "id": r.id,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "Recent",
            "record_type": r.title,
            "summary": r.description[:250] + "..." if r.description and len(r.description) > 250 else r.description,
            "patient_name": r.patient.full_name if r.patient else "Patient",
            "has_file": bool(r.file_path and os.path.exists(r.file_path)),
            "file_name": r.file_name or "Report"
        }
        for r in records
    ]
    return formatted_records


@app.get("/api/ehr/download/{record_id}")
def download_ehr_report(record_id: int, db: Session = Depends(get_db)):
    record = db.query(MedicalRecord).filter(
        MedicalRecord.id == record_id).first()
    if not record:
        raise HTTPException(
            status_code=404, detail="Medical record not found.")

    # 1. If an original uploaded file exists (JPEG, PNG, PDF), serve that file!
    if record.file_path and os.path.exists(record.file_path):
        return FileResponse(
            path=record.file_path,
            filename=record.file_name or os.path.basename(record.file_path),
            media_type="application/octet-stream"
        )

    # 2. Otherwise (for appointments/notes without physical files), generate text report
    patient_name = record.patient.full_name if record.patient else "Unknown Patient"
    date_str = record.created_at.strftime(
        "%Y-%m-%d %H:%M:%S") if record.created_at else "N/A"

    report_content = f"""====================================================================
                        MEDICORE AI CLINICAL REPORT
====================================================================
Record ID    : {record.id}
Date Generated: {date_str}
Patient Name : {patient_name}
Title/Type   : {record.title}
====================================================================

CLINICAL NOTES:
--------------------------------------------------------------------
{record.description or 'No notes recorded.'}
"""

    filename = f"MediCore_Appointment_{record.id}.txt"
    return Response(
        content=report_content,
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )