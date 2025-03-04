from src.concrete.model import ModelInferenceStep


class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def run(self, input_data=None):
        data = input_data
        for step in self.steps:
            data = step.run(data)
        return data
