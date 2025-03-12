import csv

class Writer:
    def __init__(self, output_file: str):
        self.output_file = output_file
        with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Subject", "Content"])

    def write_row(self, subject: str, content: str):
        with open(self.output_file, 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([subject, content])