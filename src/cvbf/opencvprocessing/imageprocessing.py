import uuid

class ImageProcessing():
    def __init__(self, input, DEVICE_NAME, DEVICE_INDEX) -> None:
        self.id = uuid.uuid1()
        self.input = input
        self.DEVICE_NAME = DEVICE_NAME
        self.DEVICE_INDEX = DEVICE_INDEX
        self.DEVICE = f'{self.DEVICE_NAME}:{self.DEVICE_INDEX}'
        self.output = None
