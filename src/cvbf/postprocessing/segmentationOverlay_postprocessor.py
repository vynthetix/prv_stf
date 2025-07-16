import torch
from torchvision.utils import draw_segmentation_masks
from cvbf.postprocessing.postprocessing import PostProcessing
from time import time as now

class SegmentationOverlayPostProcessor(PostProcessing):
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

    def drawMasks(self, masks, colors = None, alpha=0.8) -> None:
        """
        draws the boxes above the given threshold over the input image

        Parameters:
        -----------
        boxes : torch.Tensor
            The boxes from the inference.
        scores : torch.tensor
            The scores of the boxes.
        threshold : The minimal detection threshold required 

        Returns:
        --------
        None
        """
        start = now()
        self.output = [draw_segmentation_masks(image=self.input.output[0], masks=masks, colors=colors, alpha=alpha), now()-start]
