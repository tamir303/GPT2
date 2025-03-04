from src.interface import Step


class UserInputStep(Step):
    """
    A pipeline Step that fetches user input text. This can be from console,
    a GUI prompt, or any custom source. Returns the user text.

    If you'd prefer not to block for console input, you can adapt this logic
    to read from a file, environment variable, etc.
    """

    def __init__(self):
        pass

    def run(self, input_data=None):
        """
        :param input_data: not used, as we always ask the user for new input
        :return: the text string the user enters
        """

        if input_data is None:
            raise ValueError("UserInputStep requires input data")

        prompt = input_data
        return prompt
