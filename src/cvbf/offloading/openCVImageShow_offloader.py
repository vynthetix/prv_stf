from cvbf.offloading.offload import Offload
import numpy as np
import cv2

class OpenCVImageShowOffloader(Offload):
    """
    Offloads image processing tasks to a separate thread for real-time display.

    Parameters:
    input (object): The input object containing the image data to be displayed.

    Returns:
    None
    """
    def __init__(self, input) -> None:
        """
        Initializes the ImageShowOffloader class.

        Calls the parent class's __init__ method with the provided input.

        Parameters:
        input (object): The input object containing the image data to be displayed.

        Returns:
        None
        """
        super().__init__(input=input)
        self.key = ""

    def show(self, height, refreshrate: int = 10):
        """
        Displays the processed image in a window.

        Resizes the image to the specified height while maintaining aspect ratio,
        converts the image to RGB format, and displays it in a window.

        Parameters:
        height (int): The desired height of the displayed image.
        refreshrate (int, optional): The time interval in milliseconds between each frame display. Defaults to 10.

        Returns:
        None
        """
        img = self.input.output
        if height != None:
            img = cv2.resize(cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB), dsize=(int((height/img.shape[0])*img.shape[1]), int(height)), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("frame", img)
        self.key = cv2.waitKey(refreshrate)
