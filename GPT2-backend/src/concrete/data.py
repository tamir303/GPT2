from src.interface.step import Step
from src.impl import data_ingestion, data_preprocess, utils
from src.impl.model_ops import setup_tokenizer

class DataIngestionStep(Step):
    def __init__(self, source_path):
        self.document_processor = data_ingestion.DocumentProcessor(source_path)

    def run(self, input_data=None):
        self.document_processor.process()
        data = self.document_processor.get_content()
        path = self.document_processor.get_file_path()

        return data, path


class PreprocessingStep(Step):
    def __init__(self, config=None):
        self.config = config  # e.g., any preprocessing parameters

    def run(self, input_data=None):
        if input_data is None:
            raise ValueError("No input data provided.")

        data, path = input_data
        cleaned = data_preprocess.clean_data(data, config=self.config)

        return cleaned, path


class DataSplitStep(Step):
    def __init__(self, test_size=0.2):
        self.test_size = test_size

    def run(self, input_data=None):
        if input_data is None:
            raise ValueError("No input data provided.")

        data, path = input_data
        train, test = utils.split_train_test(data, split = 1 - self.test_size)

        return data, path, test, train


class DataTokenizer(Step):
    def __init__(self, vocab_file):
        self.vocab_file = vocab_file

    def run(self, input_data=None):
        if input_data is None:
            raise ValueError("No input data provided.")

        data, path, test, train = input_data
        tokenizer = setup_tokenizer(path)

        return tokenizer, test, train