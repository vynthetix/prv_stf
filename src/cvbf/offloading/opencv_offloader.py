from cvbf.offloading.offload import Offload
import numpy as np

class OpenCVOffloader(Offload):
    """
    Converts the input data to the OpenCV format.

    The input data is expected to be in a specific format, which is then
    permuted, moved to CPU, converted to numpy array, and scaled to the range of 0-255.
    The resulting array is then cast to unsigned 8-bit integers.

    Parameters:
    None

    Returns:
    None
    """

    def __init__(self, input) -> None:
        super().__init__(input=input)

    def convertToOpenCV(self, ):
        self.output = ((self.input.output[0]).permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
