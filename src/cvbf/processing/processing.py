import  uuid

class Processing:
    """
    This class is responsible for processing input data and performing operations.

    Attributes:
    id (UUID): A unique identifier for the processing instance.
    input (object): The input data to be processed.
    output (object): The processed output data.
    output_ready (bool): A flag indicating whether the output is ready for use.
    DEVICE (str): The device information from the input.
    DEVICE_NAME (str): The name of the device.
    DEVICE_INDEX (int): The index of the device.

    Methods:
    __init__(self, input) -> None:
        Initializes the Processing instance with the given input data.
    """
    def __init__(self, input) -> None:
        """
        Initializes the Processing instance with the given input data.

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
        self.DEVICE_NAME = self.DEVICE.split(":")[0]
        self.DEVICE_INDEX = int(self.DEVICE.split(":")[1])

