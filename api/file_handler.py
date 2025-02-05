from pathlib import Path

from PyPDF2 import PdfReader
from docx import Document


class DocumentProcessor:
    def __init__(self, input_path: str):
        self.input_path = Path(input_path)
        self.output_path = None
        self.content = None

    def process(self):
        """Process the input file and convert it to a text file."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"The file {self.input_path} does not exist.")

        file_extension = self.input_path.suffix.lower()
        if file_extension == '.pdf':
            self._process_pdf()
        elif file_extension == '.docx':
            self._process_docx()
        elif file_extension == '.txt':
            self._process_txt()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _process_pdf(self):
        """Convert a PDF file to a text file."""
        with open(self.input_path, 'rb') as file:
            reader = PdfReader(file)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            self._save_to_txt(text)

    def _process_docx(self):
        """Convert a DOCX file to a text file."""
        doc = Document(self.input_path.stem)
        text = '\n'.join([para.text for para in doc.paragraphs])
        self._save_to_txt(text)

    def _process_txt(self):
        """Read a TXT file."""
        with open(self.input_path, 'r', encoding='utf-8') as file:
            text = file.read()
        self._save_to_txt(text)

    def _save_to_txt(self, text: str):
        """Save the extracted text to a new TXT file."""
        output_dir = self.input_path.parent / 'processed_texts'
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = output_dir / f"{self.input_path.stem}.txt"
        with open(self.output_path, 'w', encoding='utf-8') as file:
            file.write(text)
        self.content = text

    def get_file_path(self):
        """Return the path of the generated text file."""
        return self.output_path

    def get_content(self):
        """Return the content of the generated text file."""
        return self.content
