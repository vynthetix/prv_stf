import uuid

class PipelineClass():
    """
    A class representing the pipeline for processing data.

    Attributes
    ----------
    id : uuid.UUID
        A unique identifier for the pipeline instance.
    input_queue : Queue
        The queue from which input data is received.
    output_queue : Queue
        The queue to which processed data is sent.
    DEVICE_NAME : str
        The name of the device used for processing.
    DEVICE_INDEX : int
        The index of the device used for processing.

    Methods
    -------
    get_id()
        Returns the unique identifier of the pipeline instance.
    """

    def __init__(self, input_queue, output_queue, DEVICE_NAME, DEVICE_INDEX) -> None:
        """
        Initializes a new instance of the PipelineClass.

        Parameters
        ----------
        input_queue : Queue
            The queue from which input data is received.
        output_queue : Queue
            The queue to which processed data is sent.
        DEVICE_NAME : str
            The name of the device used for processing.
        DEVICE_INDEX : int
            The index of the device used for processing.
        """
        self.id = uuid.uuid1()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.DEVICE_NAME = DEVICE_NAME
        self.DEVICE_INDEX = DEVICE_INDEX

    def get_id(self):
        """
        Returns the unique identifier of the pipeline instance.

        Returns
        -------
        uuid.UUID
            The unique identifier of the pipeline instance.
        """
        return self.id