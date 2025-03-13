from abc import ABC, abstractmethod


class IDataLoader(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_file_path(self) -> str:
        return self.file_path

    @abstractmethod
    def get_file_content(self) -> str:
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __del__(self):
        pass