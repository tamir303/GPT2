import shutil
from pathlib import Path

from fastapi import UploadFile, File, APIRouter

from data.file_handler import DocumentProcessor

processor: DocumentProcessor | None = None

router = APIRouter(prefix="/upload", tags=["Files"])

@router.post("/process_file/")
async def process_file(file: UploadFile = File(...)):
    global processor

    temp_file_path = Path(f"temp_files/{file.filename}")
    temp_file_path.parent.mkdir(parents=True, exist_ok=True)

    with temp_file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Now process the saved file
    processor = DocumentProcessor(str(temp_file_path))
    processor.process()

    # Get the name and content of the processed text file
    processed_file_path = processor.get_file_path()
    content = processor.get_content()

    return {
        "file_name": processed_file_path.name,
        "content": content
    }