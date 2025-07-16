"""
A script used to perform dummy processing on input data.
"""

import torch

DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE_INDEX = 0

# load the file
from cvbf.loading.image_file_loader import ImageFileLoader
image = ImageFileLoader(DEVICE_NAME=DEVICE_NAME, DEVICE_INDEX=DEVICE_INDEX, PATH_TO_IMAGE="outer_assets/samples/01.png")

#  create the pre-processing processor
from cvbf.preprocessing.dummy_preprocessor import DummyPreProcessor
dummyProcessor = DummyPreProcessor(input=image)

# offload from torch to openCV
from cvbf.offloading.opencv_offloader import OpenCVOffloader
openCVOffloader = OpenCVOffloader(input=dummyProcessor)

# display the image using openCV
from cvbf.offloading.openCVImageShow_offloader import OpenCVImageShowOffloader
openCVImageShowOffloader = OpenCVImageShowOffloader(input=openCVOffloader)

# run the pipeline
image.load()

dummyProcessor.run()

openCVOffloader.convertToOpenCV()

openCVImageShowOffloader.show(height=320, refreshrate=0) # offload the result by showing the image
