import torch
import cv2
from time import time as now
from cvbf.loading.loading import Loading

class ImageFileLoader(Loading):
    """
    A class to load and preprocess image files for further processing.

    Attributes:
    -----------
    DEVICE_NAME : str
        The name of the device to load the image onto.
    DEVICE_INDEX : int
        The index of the device to load the image onto.
    PATH_TO_IMAGE : str
        The path to the image file to be loaded.

    Methods:
    --------
    __init__(self, DEVICE_NAME, DEVICE_INDEX, PATH_TO_IMAGE)
        Initializes the ImageFileLoader object and loads the image.

    __load(self, PATH_TO_IMAGE)
        Private method to load the image file and preprocess it.
    """
    def __init__(self, DEVICE_NAME, DEVICE_INDEX, PATH_TO_IMAGE) -> None:
        """
        Initializes the ImageFileLoader object and loads the image.

        Parameters:
        -----------
        DEVICE_NAME : str
            The name of the device to load the image onto.
        DEVICE_INDEX : int
            The index of the device to load the image onto.
        PATH_TO_IMAGE : str
            The path to the image file to be loaded.
        """
        super().__init__(DEVICE_NAME=DEVICE_NAME, DEVICE_INDEX=DEVICE_INDEX)
        self.PATH_TO_IMAGE = PATH_TO_IMAGE
        self.lock(self.id)
        self.output = self.get_data(self.id)
        self.unlock(self.id)

    def load(self) -> None:
        """
        Private method to load the image file and preprocess it.

        Parameters:
        -----------
        PATH_TO_IMAGE : str
            The path to the image file to be loaded.

        Returns:
        --------
        None
        """
        self.lock(self.id)
        img = torch.from_numpy(cv2.imread(self.PATH_TO_IMAGE)[:,:,[2,1,0]]/255.0).to(self.DEVICE).permute(2,0,1)
        self.set_data(img,
                      0,
                      now(),
                      self.id)
        self.unlock(self.id)