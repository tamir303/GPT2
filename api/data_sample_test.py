from functools import lru_cache


@lru_cache()
def load_data() -> str:
    with open("data/ClimateChangeAnalysis.txt", "rt") as file:
        return file.read()
