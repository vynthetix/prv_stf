import torch
from torchvision.utils import draw_bounding_boxes
from cvbf.postprocessing.postprocessing import PostProcessing
from time import time as now

class BoxOverlayPostProcessor(PostProcessing):
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

    def drawBoxes(self, boxes, scores, threshold = None, topN = None, colors = (0,1,0), width = 1, order=[1,0,3,2]) -> None:
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
        if topN is None:
            boxes = boxes[0:torch.max(torch.where(scores>threshold)[0])+1,...]
        else:
            boxes = boxes[0:topN,...]
        boxes = boxes[...,order]  * torch.tensor([self.input.output[0].shape[2],self.input.output[0].shape[1],self.input.output[0].shape[2],self.input.output[0].shape[1]]).to(self.DEVICE)
        self.output = [draw_bounding_boxes(image=self.input.output[0], boxes=boxes, width=width, colors=colors), now()-start]
