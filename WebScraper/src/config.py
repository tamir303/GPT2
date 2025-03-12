class Config:
    def __init__(self, start_url: str, output_file: str, verify_ssl: bool = True, max_workers: int = 5, max_depth: int = -1, max_pages: int = -1):
        self.start_url = start_url
        self.output_file = output_file
        self.verify_ssl = verify_ssl
        self.max_workers = max_workers
        self.max_depth = max_depth  # -1 for no limit
        self.max_pages = max_pages  # -1 for no limit