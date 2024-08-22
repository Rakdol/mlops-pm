from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.db import cruds, schemas
from src.db.database import get_db

router = APIRouter()


@router.get("/projects/all")
def project_all(db: Session = Depends(get_db)):
    return cruds.select_project_all(db=db)
