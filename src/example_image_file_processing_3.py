"""
    A script to demonstrate the offloading of a preprocessed image to the specified path by saving it as an image file.
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

from cvbf.offloading.image_save_offloader import ImageSaveOffloader
offloader = ImageSaveOffloader(input=dummyProcessor)

image.load()
dummyProcessor.run()
offloader.save("output.png") # offload the result by saving the image
