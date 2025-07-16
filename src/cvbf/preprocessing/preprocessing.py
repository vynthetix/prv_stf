import uuid

class PreProcessing:
    """
    This class is responsible for pre-processing input data.

    Attributes:
    -----------
    id : uuid.UUID
        A unique identifier for the instance.
    input : InputObject
        The input object to be processed.
    output : Any
        The processed output data.
    output_ready : bool
        A flag indicating whether the output data is ready.
    DEVICE : str
        The device used for processing.

    Methods:
    --------
    update_data()
        Updates the input data and prepares the output.
    """
    def __init__(self, input) -> None:
        """
        Initializes a new instance of PreProcessing.

        Parameters:
        -----------
        input : InputObject
            The input object to be processed.
        """
        self.id = uuid.uuid1()
        self.input = input
        self.output = None
        self.output_ready = False
        self.DEVICE = input.DEVICE

