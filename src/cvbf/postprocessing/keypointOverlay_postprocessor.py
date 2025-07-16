import torch
from torchvision.utils import draw_keypoints
from cvbf.postprocessing.postprocessing import PostProcessing
from time import time as now

class KeypointOverlayPostProcessor(PostProcessing):
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

    def drawKeypoints(self, keypoints, scores, threshold = None, topN = 1, colors = (0,1,0), radius = 1, order=[1,0]) -> None:
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
        selection = torch.unique(torch.where(scores>threshold)[0])[0:topN,...]
        keypoints = keypoints[selection,...]
        keypoints = keypoints[...,order]  * (torch.tensor([self.input.output[0].shape[2],self.input.output[0].shape[1]]).to(self.DEVICE)).repeat(keypoints.shape[len(keypoints.shape)-2], 1)
        scores = scores[selection,...]
        self.output = [draw_keypoints(image=self.input.output[0], keypoints=keypoints, radius=radius, colors=colors), now()-start]
