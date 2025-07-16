import time
from cvbf.loading.loading import Loading

class ImageObjectLoader(Loading):
    """
    A class to load image objects into a device for further processing.

    Attributes
    ----------
    DEVICE_NAME : str
        The name of the device to load the image onto.
    DEVICE_INDEX : int
        The index of the device to load the image onto.

    Methods
    -------
    load(image)
        Load the given image onto the device.

    """
    def __init__(self, DEVICE_NAME, DEVICE_INDEX) -> None:
        """
        Initialize the ImageObjectLoader with the given device name and index.

        Parameters
        ----------
        DEVICE_NAME : str
            The name of the device to load the image onto.
        DEVICE_INDEX : int
            The index of the device to load the image onto.

        """
        super().__init__(DEVICE_NAME=DEVICE_NAME, DEVICE_INDEX=DEVICE_INDEX)

    def load(self, image) -> None:
        """
        Load the given image onto the device.

        Parameters
        ----------
        image : torch.Tensor
            The image to load onto the device.

        Returns
        -------
        None
            The function does not return any value.

        """
        self.lock(self.id)
        self.set_data(image.to(self.DEVICE),
                      0,
                      time.time(),
                      self.id)
        self.unlock(self.id)

