"""
This script represents a complete image processing pipeline for anomaly detection.
It includes loading an image, pre-processing patches, performing inference using an ONNX model,
post-processing the results, overlaying the original image with the activity map, and showing the final image.

Parameters:
    - path_to_image (str): The path to the image file to be processed.
    - path_to_model_dir (str): The directory path containing the ONNX model files.
    - device_name (str): The name of the device to use for processing (default is 'cuda' if available, else 'cpu').
    - device_index (int): The index of the device to use for processing (default is 0).
    - height (int): The height of the displayed image (default is 320).
    - refreshrate (int): The refresh rate of the displayed image (-1 for continuous, default is -1).     
"""

import torch

DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE_INDEX = 0

# load the file
from cvbf.loading.image_file_loader import ImageFileLoader
image = ImageFileLoader(DEVICE_NAME=DEVICE_NAME, DEVICE_INDEX=DEVICE_INDEX, PATH_TO_IMAGE="/app/outer_assets/samples/01.png")

#  create the pre-processing processor
from cvbf.preprocessing.patch_preprocessor import PatchPreProcessor
patches = PatchPreProcessor(input=image)

# create the inference processor
from cvbf.processing.onnx_processor import OnnxProcessor
inference = OnnxProcessor(input=patches, PATH_TO_MODEL_DIR="/app/outer_assets/model/anomaly_detection/")

# create the post-processing processor
from cvbf.postprocessing.patch_processor import PatchPostProcessor
result = PatchPostProcessor(input=inference)

# add the original image to the result
from cvbf.postprocessing.imageOverlay_postprocessor import ImageOverlayPostProcessor
overlay = ImageOverlayPostProcessor(input=result)

# offload from torch to openCV
from cvbf.offloading.opencv_offloader import OpenCVOffloader
openCVOffloader = OpenCVOffloader(input=overlay)

# display the image using openCV
from cvbf.offloading.openCVImageShow_offloader import OpenCVImageShowOffloader
openCVImageShowOffloader = OpenCVImageShowOffloader(input=openCVOffloader)

# run the pipeline
inference.load_model()
inference.load_metadata()

image.load()

patches.create(384)

inference.run_with_iobinding() # perform processing

result.create_activity_map(cell_size=48, threshold=0.99, resultIndex=0) # perform post-processing

overlay.underlay(underlay=image.output[0], alpha=0.8) # overlay the original image and the activity map

openCVOffloader.convertToOpenCV()

openCVImageShowOffloader.show(height=320, refreshrate=-1) # offload the result by showing the image
