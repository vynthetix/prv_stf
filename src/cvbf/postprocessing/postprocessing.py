import uuid

class PostProcessing:
    """
    This class is responsible for performing post-processing operations on input data.

    Attributes:
    id (UUID): A unique identifier for the post-processing instance.
    input (object): The input data to be processed.
    output (object): The result of the post-processing operation.
    output_ready (bool): A flag indicating whether the output is ready for use.
    DEVICE (object): The device on which the input data is processed.

    Methods:
    __init__(self, input) -> None:
        Initializes a new instance of PostProcessing with the given input data.
    """

    def __init__(self, input) -> None:
        """
        Initializes a new instance of PostProcessing with the given input data.

        Parameters:
        input (object): The input data to be processed.

        Returns:
        None
        """
        self.id = uuid.uuid1()
        self.input = input
        self.output = None
        self.output_ready = False
        self.DEVICE = input.DEVICE