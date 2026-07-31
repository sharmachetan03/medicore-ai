from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(50), default="patient")  # 'patient', 'doctor', 'admin'
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class MedicalRecord(Base):
    __tablename__ = "medical_records"

    id = Column(Integer, primary_key=True, index=True)
    user_email = Column(String(255), index=True, nullable=False)
    record_type = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Appointment(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    patient_email = Column(String(255), nullable=False)
    doctor_email = Column(String(255), nullable=False)
    appointment_date = Column(String(100), nullable=False)
    reason = Column(Text, nullable=False)
    status = Column(String(50), default="scheduled")
    created_at = Column(DateTime(timezone=True), server_default=func.now())