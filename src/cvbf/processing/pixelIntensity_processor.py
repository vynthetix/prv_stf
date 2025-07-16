import torch
import cv2
from cvbf.processing.processing import Processing

class PixelIntensityProcessor(Processing):
    def __init__(self, input):
        """
        Initialize the PixelIntensityProcessor class with the given input.

        Parameters:
        input (object): The input data for processing.

        Attributes:
        zeros (torch.Tensor): A tensor of zeros with a single element, moved to the device specified by self.DEVICE.
        ones (torch.Tensor): A tensor of ones with a single element, moved to the device specified by self.DEVICE.
        cell_flag (torch.Tensor): A tensor to store flags indicating whether a cell is flagged as an anomaly or not. Initialized as None.
        cell_vault (torch.Tensor): A tensor to store the patches of cells for comparison. Initialized as None.
        cell_vault_active (torch.Tensor): A tensor to store the active status of the cell vault. Initialized as None.
        mask (torch.Tensor): A tensor to store the mask indicating the active pixels in a cell. Initialized as None.
        notification (int): An integer to store the notification status. Initialized as 0.
        """
        super().__init__(input=input)
        self.zeros = torch.zeros((1)).to(self.DEVICE)
        self.ones = torch.ones((1)).to(self.DEVICE)
        self.cell_flag = None
        self.cell_vault = None
        self.cell_vault_active = None
        self.mask = None
        self.notification = 0

    def _normalize_patch(self, patch):
        """
        Normalize the input patch by scaling its values to the range [0, 1].

        Parameters:
        patch (torch.Tensor): The input patch to be normalized. The patch should have shape (C, H, W), where C is the number of channels, H is the height, and W is the width.

        Returns:
        torch.Tensor: The normalized patch with values in the range [0, 1]. The shape of the output tensor is the same as the input tensor.
        """
        patch_min, _ = torch.min(patch, dim=1, keepdim=True)
        patch_max, _ = torch.max(patch, dim=1, keepdim=True)
        return (patch - patch_min) / (patch_max - patch_min)


    def run(self, min_required_active_pixels_percent, noise_reduction_intensity, sensitivety_to_movement):
        """
        This function processes the input data to identify anomalies in the cells. It normalizes the input patches,
        calculates the mask indicating active pixels, and updates the cell flags based on the specified parameters.
        It also checks for movement in consecutive frames and updates the cell vault accordingly.

        Parameters:
        min_required_active_pixels_percent (float): The minimum percentage of active pixels required in a cell to consider it as an anomaly.
        noise_reduction_intensity (float): The intensity threshold for noise reduction. Pixels with intensity below this threshold are considered noise.
        sensitivety_to_movement (float): The sensitivity threshold for detecting movement between consecutive frames. If the similarity value between two frames is above this threshold, a movement is detected.

        Returns:
        None. The function updates the attributes of the PixelIntensityProcessor class.
        """
        self.output = self.input.output

        if self.cell_flag == None:
            self.cell_flag_idx = 0
            self.cell_flag = torch.zeros(len(self.input.output)).to(self.DEVICE)
            self.cell_vault = torch.zeros(len(self.input.output), self.input.output[0].shape[0], self.input.output[0].shape[1], self.input.output[0].shape[2]).to(self.DEVICE)
            self.cell_vault_error = torch.zeros(len(self.input.output))
            self.cell_vault_error[:] = -1
            self.cell_vault_active = torch.zeros(len(self.input.output)).to(self.DEVICE)
            self.mask = torch.zeros(len(self.input.output), 1, self.input.output[0].shape[1], self.input.output[0].shape[2]).to(self.DEVICE)

        for i in range(0, len(self.input.foreground)):
            self.input.foreground[i] = self._normalize_patch(torch.clamp(self.input.foreground[i], min=noise_reduction_intensity, max=1.0))
            self.mask[i][0,:] = torch.ceil(torch.clamp((self.input.foreground[i][0,:]+self.input.foreground[i][1,:]+self.input.foreground[i][2,:])/3.0 - noise_reduction_intensity, min=0.0, max=1.0))

            if torch.sum(self.mask[i])/(self.mask[i].shape[1]*self.mask[i].shape[2]) > min_required_active_pixels_percent:
                self.cell_flag[i] = 1.0
            else:
                self.cell_flag[i] = 0.0

        self.notification = 0
        summary = [cell == 1 for cell in self.cell_flag]
        xmin = int(self.input.output[0].shape[1]/7)
        xmax = self.input.output[0].shape[1]-xmin
        ymin = int(self.input.output[0].shape[2]/7)
        ymax = self.input.output[0].shape[2]-ymin
        for i in range(0, len(summary)):
            if summary[i] == True:
                #Check if the previous patch was identified as an anomaly as well.
                if self.cell_vault_active[i] == 1:
                    # store the difference between two consequtive patches in the vault.
                    error = self._estimate_siliarity(patch = self.input.output[i], template = self.cell_vault[i][:,ymin:ymax,xmin:xmax])
                    self.cell_vault_error[i] = abs(error[0]-ymin)+abs(error[1]-xmin)
                    if self.cell_vault_error[i] <= sensitivety_to_movement:
                        # if the calculated similarity value is among the 90
                        self.notification = 1
                        self.output[i][0,:] = 1.0
                    else:
                        # if a jam was detected, we do not update the cell vault to test if the next frame still corresponds to the frame we are looking at.
                        self.cell_vault[i] = self.input.output[i]
                        self.cell_vault_active[i] = 0.0
                else:
                    self.cell_vault_active[i] = 1.0
                    self.cell_vault[i] = self.input.output[i]
            else:
                self.cell_vault_active[i] = 0.0


    def _minMaxLoc(self, tensor):
        # not working right now ...
        values_1, indices_1 = tensor.max(dim=1, keepdim=True)
        values_2, indices_2 = tensor.max(dim=0, keepdim=True)
        #return ((min(indices_1[0]), min(indices_2[0])), (max(indices_1[0]), max(indices_2[0])))
        return None

    def _convert_rgb_to_grayscale(self, img):
        """
        Converts an RGB image to grayscale using the luminosity method.
        https://github.com/pytorch/vision/blob/main/torchvision/transforms/_functional_tensor.py#L157-L161#-

        Parameters:
        img (torch.Tensor): The input RGB image tensor. The tensor should have shape (C, H, W), where C is the number of channels (3 for RGB), H is the height, and W is the width.

        Returns:
        torch.Tensor: The grayscale image tensor. The tensor has shape (1, H, W), where 1 represents the single channel for grayscale images.
        """
        r, g, b = img.unbind(dim=-3)
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
        return l_img.unsqueeze(dim=-3)

    def _tm_ccorr_normed(self, patch, template):
        """
        This function performs normalized cross-correlation using convolution for two input tensors.
        https://github.com/hirune924/TemplateMatching/blob/master/Template%20Matching%20(PyTorch%20implementation).ipynb

        Parameters:
        patch (torch.Tensor): The first input tensor representing the patch. It should have shape (C, H, W), where C is the number of channels, H is the height, and W is the width.
        template (torch.Tensor): The second input tensor representing the template. It should have the same shape as the patch.

        Returns:
        torch.Tensor: The normalized cross-correlation result. The output tensor has shape (H, W), representing the correlation between the patch and the template.
        """
        patch = patch.unsqueeze(dim=0).type(torch.float32)
        template = template.unsqueeze(dim=0).expand(1, template.shape[0], template.shape[1], template.shape[2]).type(torch.float32)
        result1 = torch.nn.functional.conv2d(patch, template, bias=None, stride=1, padding=0)
        result2 = torch.sqrt(torch.sum(template**2) * torch.nn.functional.conv2d(patch**2, torch.ones_like(template), bias=None, stride=1, padding=0))
        return (result1/result2).squeeze(0).squeeze(0)


    def _estimate_siliarity(self, patch, template):
        """
        Estimates the similarity between two input patches using normalized cross-correlation.

        Parameters:
        patch (torch.Tensor): The first input patch tensor. It should have shape (C, H, W), where C is the number of channels, H is the height, and W is the width.
        template (torch.Tensor): The second input template tensor. It should have the same shape as the patch.

        Returns:
        tuple: A tuple containing the coordinates (x, y) of the maximum correlation value in the resulting normalized cross-correlation map.
        """
        patch = self._convert_rgb_to_grayscale(img=patch)
        template = self._convert_rgb_to_grayscale(img=template)
        result = self._tm_ccorr_normed(patch=patch, template=template).cpu().numpy()
        _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
        return maxLoc