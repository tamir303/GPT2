from functools import lru_cache

def get_file_path():
    return "data/ClimateChangeAnalysis.txt"

@lru_cache()
def get_content() -> str:
    with open(get_file_path(), "rt") as file:
        return file.read()


