import uuid

class Offload:
    """
    Offload class for managing data offloading operations.

    Attributes:
    id (UUID): Unique identifier for the offload operation.
    input (object): Input data for the offload operation.
    output (object): Output data after the offload operation.
    output_ready (bool): Flag indicating if the output data is ready.
    DEVICE (str): Device information for the offload operation.
    DEVICE_NAME (str): Name of the device for the offload operation.
    DEVICE_INDEX (int): Index of the device for the offload operation.

    Methods:
    __init__(self, input) -> None:
        Constructor method to initialize the Offload object with input data.
    """
    def __init__(self, input) -> None:
        """
        Initialize Offload object with input data.

        Parameters:
        input (object): Input data for the offload operation.

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

