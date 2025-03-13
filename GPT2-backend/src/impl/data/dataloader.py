import csv
import json
import os
from typing import Union, List

from src.interfaces.dataloader import IDataLoader
import mmap

class DataLoader(IDataLoader):
    def __init__(self, file_path: str):
        super().__init__(file_path)
        self.file_ext = os.path.splitext(file_path)[1].lower()  # Get file extension

        self.file = open(file_path, 'r', encoding="utf-8")
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

        content = self.mmap.read().decode('utf-8')

        if self.file_ext == ".txt":
            # For plain text, use content as is.
            self.data = content

        elif self.file_ext == ".json":
            try:
                # Parse the JSON and then dump it back to a string
                json_obj = json.loads(content)
                self.data = json.dumps(json_obj, indent=2)
            except RuntimeError as e:
                # Fallback: if parsing fails, keep the raw content
                self.data = content

        elif self.file_ext == ".csv":
            try:
                # Parse CSV rows and join them into a string
                reader = csv.reader(content.splitlines())
                rows = list(reader)
                # Join each row into a comma-separated string, then join rows with newlines
                self.data = "\n".join([",".join(row) for row in rows])
            except RuntimeError as e:
                # Fallback: if parsing fails, use raw content
                self.data = content

        else:
            raise ValueError("Unsupported file format. Use .txt, .json, or .csv")

    def get_file_content(self) -> str:
        return self.data

    def __len__(self):
        return len(self.data)

    def __del__(self):
        self.mmap.close()
        self.file.close()