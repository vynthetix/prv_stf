import torch
import torchvision.transforms.functional as F
from cvbf.postprocessing.postprocessing import PostProcessing
from time import time as now

class ImageOverlayPostProcessor(PostProcessing):
    """
    A class for performing overlay and underlay operations on input data.

    Attributes:
    -----------
    input : torch.Tensor
        The input data on which overlay and underlay operations will be performed.

    Methods:
    --------
    overlay(overlay, alpha, interpolationMode)
        Overlays the given overlay image on the input data.

    underlay(underlay, alpha, interpolationMode)
        Underlays the given underlay image on the input data.
    """
    def __init__(self, input):
        """
        Initializes the OverlayPostProcessor class with the given input data.

        Parameters:
        -----------
        input : torch.Tensor
            The input data on which overlay and underlay operations will be performed.
        """
        super().__init__(input=input)

    def overlay(self, overlay, alpha, interpolationMode: F.InterpolationMode = F.InterpolationMode.BILINEAR) -> None:
        """
        Overlays the given overlay image on the input data.

        Parameters:
        -----------
        overlay : torch.Tensor
            The overlay image to be added on the input data.
        alpha : float
            The transparency factor for the overlay image.
        interpolationMode : torchvision.transforms.functional.InterpolationMode, optional
            The interpolation mode to be used for resizing the overlay image.
            Default is F.InterpolationMode.BILINEAR.

        Returns:
        --------
        None
        """
        overlay = F.resize(img=overlay, interpolation=interpolationMode, size=(self.input.output[0].shape[1],self.input.output[0].shape[2]))
        self.output = torch.add(input=(self.input.output[0]), other=overlay, alpha=alpha).unsqueeze(0)

    def underlay(self, underlay, alpha, interpolationMode: F.InterpolationMode = F.InterpolationMode.BILINEAR) -> None:
        """
        Underlays the given underlay image under the input data.

        Parameters:
        -----------
        underlay : torch.Tensor
            The underlay image to be added on the input data.
        alpha : float
            The transparency factor for the underlay image.
        interpolationMode : torchvision.transforms.functional.InterpolationMode, optional
            The interpolation mode to be used for resizing the underlay image.
            Default is F.InterpolationMode.BILINEAR.

        Returns:
        --------
        None
        """
        start = now()
        self.output = F.resize(img=self.input.output[0], interpolation=interpolationMode, size=(underlay.shape[1],underlay.shape[2]))
        self.output = self.output.repeat(underlay.shape[0], 1, 1)
        self.output = [torch.add(input=underlay, other=self.output[0], alpha=alpha), now()-start]
