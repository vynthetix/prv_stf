from cvbf.preprocessing.preprocessing import PreProcessing
from time import time as now

class DummyPreProcessor(PreProcessing):
    """
    A class to process input data for a computer vision task.

    Attributes
    ----------
    input : object
        The input data to be processed.

    Methods
    -------
    __init__(self, input)
        Initializes the DummyProcessor with the given input.

    run()
        Processes the input data and generates the output.
    """

    def __init__(self, input) -> None:
        """
        Initializes the DummyProcessor with the given input.

        Parameters
        ----------
        input : object
            The input data to be processed.
        """
        super().__init__(input=input)

    def run(self):
        """
        Processes the input data and generates the output.

        The output is a tensor with an additional dimension,
        representing a batch of single frames.

        Returns
        -------
        None
            The method does not return a value, but it updates the `output` attribute.
        """
        start = now()
        self.output = [self.input.output[0], now()-start]