import torch
from cvbf.postprocessing.postprocessing import PostProcessing
from time import time as now

class PatchPostProcessor(PostProcessing):

    def __init__(self, input):
        """
        Initialize PatchPostProcessor instance.

        Parameters:
        input (object): The input data for post-processing.

        Attributes:
        zeros (torch.Tensor): A tensor filled with zeros, used for comparison.
        ones (torch.Tensor): A tensor filled with ones, used for comparison.
        """
        super().__init__(input=input)
        self.zeros = torch.zeros((1,1,1,1)).to(self.DEVICE)
        self.ones = torch.ones((1,1,1,1)).to(self.DEVICE)



    def __create_activity_grid(self, patch, size) -> torch.Tensor:
        """
        Create an activity grid from the input patch by dividing it into smaller patches of the given size.

        Parameters:
        patch (torch.Tensor): The input patch tensor with shape (C, H, W), where C is the number of channels,
                              H is the height, and W is the width.
        size (int): The size of each smaller patch.

        Returns:
        torch.Tensor: A tensor representing the activity grid. The shape of the output tensor is
                      (nbr_rows, nbr_cols, C, size, size), where nbr_rows and nbr_cols are the number
                      of rows and columns in the activity grid, respectively.
        """
        C, H, W = patch.shape
        output = patch.unfold(1, size, size).unfold(2, size, size)
        output = output.swapaxes(0,1).swapaxes(1,2)
        patches_dimension = output.shape
        return patches_dimension[0], patches_dimension[1], output.reshape(-1, C, size, size)


    def stitch_patches(self, patches: torch.Tensor, nbr_rows: int, nbr_cols: int) -> torch.Tensor:
        """
        Stitches together smaller patches into a larger image.
        Test

        Parameters:
        patches (torch.Tensor): A tensor representing the smaller patches. The shape of the tensor is
                               (nbr_patches, patch_size, patch_size, channels).
        nbr_rows (int): The number of rows in the larger image.
        nbr_cols (int): The number of columns in the larger image.

        Returns:
        torch.Tensor: A tensor representing the larger image. The shape of the tensor is
                      (1, nbr_rows*patch_size, nbr_cols*patch_size, channels).
        """
        start = now()
        patches = patches.permute(0,2,3,1)
        patch_size = patches.shape[2]
        channels = patches.shape[3]
        patches = patches.view(nbr_rows, nbr_cols, patch_size, patch_size, channels).swapaxes(2, 1)
        patches = patches.reshape(1, patch_size*nbr_rows, patch_size*nbr_cols, channels)
        self.output = [patches.permute(0,3,1,2)[0,...], now()-start]

    def _stitch_patches(self, patches: torch.Tensor, nbr_rows: int, nbr_cols: int) -> torch.Tensor:
        self.stitch_patches(patches=patches, nbr_cols=nbr_cols, nbr_rows=nbr_rows)
        return self.output[0]

    def create_activity_map(self, cell_size: int, threshold: float, resultIndex: int) -> None:
        """
        This function creates an activity map from the input data by summarizing each patch into a smaller grid.

        Parameters:
        cell_size (int): The size of each cell in the smaller grid.
        threshold (float): The threshold value for summarizing the grid cells.

        Returns:
        None: The function modifies the instance attribute 'self.output' with the created activity map.
        """
        start = now()
        def summarize_patch(patch, threshold):
            row, col, grid_cells = self.__create_activity_grid(patch, cell_size) # (256, 256, 1) -> 4, 4, (16, 64, 64, 1)
            grid_cells = torch.concatenate([torch.where(torch.sum(grid_cell >= self.input.metadata.get("threshold")) >= threshold, self.ones, self.zeros) for grid_cell in grid_cells.swapaxes(1,2).swapaxes(2,3)]) # (16, 64, 64, 1) -> (16, 1, 1, 1) # see https://github.com/openvinotoolkit/anomalib/blob/main/src/anomalib/deploy/inferencers/torch_inferencer.py
            patch = self._stitch_patches(grid_cells, row, col) # (16, 1, 1, 1) -> (1, 4, 4, 1)
            return patch
        output = torch.concatenate([summarize_patch(patch, threshold) for patch in self.input.output[0][resultIndex]]).unsqueeze(1) # (24, 256, 256, 1) --> (24, 4, 4, 1)
        self.stitch_patches(output, self.input.input.nbr_rows, self.input.input.nbr_cols)
        self.output[1] = now()-start
