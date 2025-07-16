import torch
from cvbf.preprocessing.preprocessing import PreProcessing
from time import time as now

class PatchPreProcessor(PreProcessing):
    def __init__(self, input) -> None:
        """
        Initialize PatchPreProcessor with input data and initialize necessary variables.

        Parameters:
        input (torch.Tensor): Input data for preprocessing.

        Returns:
        None
        """
        super().__init__(input=input)
        self.size = None
        self.nbr_rows = None
        self.nbr_cols = None
        self.zeros = torch.zeros((1,1,1,1)).to(self.DEVICE)
        self.ones = torch.ones((1,1,1,1)).to(self.DEVICE)


    def create(self, size) -> None:
        """
        Create patches from the input frame.

        This function divides the input frame into smaller patches of specified size.
        If the size parameter is not None, it updates the size attribute of the class.
        Then, it performs the following steps:
        1. Unfold the frame along the height and width dimensions to create patches.
        2. Swap the axes of the unfolded patches.
        3. Reshape the patches to a 4D tensor.
        4. Update the number of rows and columns of the patches.

        Parameters:
        size (int): The size of each patch. If None, the size attribute remains unchanged.

        Returns:
        None
        """
        if self.input.output[0] is not None:
            if size != None:
                self.size = size
            C, H, W = self.input.output[0].shape
            self.output = self.input.output[0].unfold(1, self.size, self.size).unfold(2, self.size, self.size)
            self.output = self.output.swapaxes(0,1).swapaxes(1,2)
            patches_dimension = self.output.shape
            self.output = self.output.reshape(-1, C, self.size, self.size).unsqueeze(0)
            self.nbr_rows = patches_dimension[0]
            self.nbr_cols = patches_dimension[1]


    def stitch(self) -> None:
        """
        Stitch the patches back into a single frame.

        This function reshapes and swaps the axes of the patches to recreate the original frame.
        The patches are assumed to have been created using the create() method.

        Parameters:
        None

        Returns:
        None

        The function updates the self.output attribute with the stitched frame.
        """
        self.output = self.output.reshape(self.nbr_rows, self.nbr_cols, self.size, self.size,1).swapaxes(2, 1)
        self.output = self.output.reshape(1, self.size*self.nbr_rows, self.size*self.nbr_cols, 1)


    def select(self, indices) -> None:
        """
        Select patches from the current output based on the provided indices.

        This function uses PyTorch's index_select method to select specific patches from the current output tensor.
        The selected patches are stored in the self.output attribute.

        Parameters:
        indices (torch.Tensor): A 1D tensor containing the indices of the patches to be selected. The indices should be of the same device as the self.DEVICE.

        Returns:
        None

        The function updates the self.output attribute with the selected patches.
        """
        self.output = torch.index_select(self.output, 0, indices.to(self.DEVICE))