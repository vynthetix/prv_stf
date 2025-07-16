from cvbf.processing.processing import Processing

class DummyProcessor(Processing):
    """
    A class to process input data and produce an output.

    Attributes
    ----------
    input : object
        The input data to be processed.

    Methods
    -------
    run()
        Processes the input data and generates the output.

    """

    def __init__(self, input):
        """
        Constructs all the necessary attributes for the DummyProcessor object.

        Parameters
        ----------
        input : object
            The input data to be processed. This should be an instance of a class that inherits from Processing.

        """
        super().__init__(input=input)

    def run(self):
        """
        Processes the input data and generates the output.

        The output is a copy of the input data.

        Returns
        -------
        None
            The function does not return anything, but it modifies the 'output' attribute of the DummyProcessor object.

        """
        self.output = self.input.output
