import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "data.txt")

def get_file_path() -> str:
    return FILE_PATH