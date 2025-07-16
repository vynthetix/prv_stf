from cvbf.preprocessing.preprocessing import PreProcessing
from time import time as now
from torchvision.transforms.functional import resize
import torch

class CutAreaPreprocessor(PreProcessing):
    def __init__(self, input) -> None:
        super().__init__(input=input)

    def resize(self, image, resizeTo):
        if resizeTo is None:
            return image
        else:
            return resize(img=image, size=resizeTo).unsqueeze(0)

    def pad(self, image, padTo):
        if padTo is None:
            return image
        else:
            img = torch.zeros(image.shape[0],padTo[0],padTo[1]).to(self.DEVICE)
            img[0:image.shape[0],0:min((image.shape[1],padTo[0])),0:min((image.shape[2],padTo[1]))] = image[:,0:min((image.shape[1],padTo[0])),0:min((image.shape[2],padTo[1]))]
            return img.unsqueeze(0)

    def run(self, channels=None, widths=None, heights=None, resizeTo=None, padTo=None):
        """
        This function performs preprocessing on an input image by cutting a specific area.

        Parameters:
        - channels (tuple, optional): A tuple specifying the start and end indices for channel selection. If None, all channels are used.
        - width (list of tuples, optional): A list of tuples specifying the start and end indices for width selection. If None, the entire width is used.
        - height (list of tuples, optional): A list of tuples specifying the start and end indices for height selection. If None, the entire height is used.

        Returns:
        - self.output (torch.Tensor): The preprocessed image after cutting the specified area, with a batch dimension added.
        """
        start = now()

        if widths is None and heights is None:
            cuts = None
        elif heights is None:
            cuts = len(widths)
        else:
            cuts = len(heights)

        if channels is None:
            self.output = [self.input.output[0]]
        else:
            self.output = [self.input.output[0][channels[0]:channels[1],:,:]]

        images = []
        if cuts is not None:
            if len(widths) != len(heights):
                print("widths and heights have different lengths")
                self.output = None
            else:
                if widths is None:
                    for i in range(0, cuts):
                        height = heights[i]
                        img = self.resize(self.output[0][:,height[0]:height[1],:], resizeTo)
                        img = self.pad(self.output[0][:,height[0]:height[1],:], padTo)
                        images.append(img)
                elif heights is None:
                    for i in range(0, cuts):
                        width = widths[i]
                        img = self.resize(self.output[0][:,:,width[0]:width[1]], resizeTo)
                        img = self.pad(self.output[0][:,:,width[0]:width[1]], padTo)
                        images.append(img)
                else:
                    for i in range(0, cuts):
                        height = heights[i]
                        width = widths[i]
                        img = self.resize(self.output[0][:,height[0]:height[1],width[0]:width[1]], resizeTo)
                        img = self.pad(self.output[0][:,height[0]:height[1],width[0]:width[1]], padTo)
                        images.append(img)

        if resizeTo is None and padTo is None:
            if len(images) > 0:
                self.output = [images, now()-start]
        else:
            self.output = [torch.cat(images, 0), now()-start]
