from fastapi import APIRouter, UploadFile, File
from app.core.rag import rebuild_vector_store
import shutil, os

router = APIRouter()

@router.post("/upload")
def upload(file: UploadFile = File(...)):
    os.makedirs("data", exist_ok=True)

    with open(f"data/{file.filename}", "wb") as f:
        shutil.copyfileobj(file.file, f)

    rebuild_vector_store()
    return {"message": "Upload & rebuild vector thành công"}
