import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "data.txt")

def load_text_file() -> str:
    with open(FILE_PATH, 'r', encoding="utf-8") as file:
        return file.read()

def get_file_path() -> str:
    return FILE_PATH