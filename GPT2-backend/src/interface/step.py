class Step:
    """Abstract base class for pipeline steps."""
    def run(self, input_data=None):
        raise NotImplementedError("Each step must implement the run() method.")