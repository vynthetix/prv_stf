import torch
import torchvision
from cvbf.offloading.offload import Offload

class ImageSaveOffloader(Offload):
    """
    Offloads the input image to a specified file path using PyTorch's torchvision library.

    Parameters:
    input (torch.Tensor): The input image tensor to be offloaded.

    Returns:
    None
    """
    def __init__(self, input) -> None:
        """
        Initializes the ImageSaveOffloader instance with the given input tensor.

        Parameters:
        input (torch.Tensor): The input image tensor to be offloaded.
        """
        super().__init__(input=input)

    def save(self, IMAGE_FILE_PATH):
        """
        Saves the input image tensor to the specified file path in PNG format.

        Parameters:
        IMAGE_FILE_PATH (str): The file path where the image will be saved.

        Returns:
        None
        """
        torchvision.io.write_png(input=(self.input.output[0].cpu()*255).type(torch.uint8), filename=IMAGE_FILE_PATH)

