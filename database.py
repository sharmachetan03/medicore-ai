from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# MySQL Connection String: mysql+pymysql://<user>:<password>@<host>/<database>
SQLALCHEMY_DATABASE_URL = "mysql+pymysql://root:Chicu2016@localhost/healthcare_db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_recycle=3600,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()