import torch

DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE_INDEX = 0

# load the file
from cvbf.loading.image_file_loader import ImageFileLoader
image_object = ImageFileLoader(DEVICE_NAME=DEVICE_NAME, DEVICE_INDEX=DEVICE_INDEX, PATH_TO_IMAGE="/app/outer_assets/samples/01.png")

# load the file
from cvbf.loading.image_object_loader import ImageObjectLoader
image = ImageObjectLoader(DEVICE_NAME=DEVICE_NAME, DEVICE_INDEX=DEVICE_INDEX)

#  create the pre-processing processor
from cvbf.preprocessing.patch_preprocessor import PatchPreProcessor
patches = PatchPreProcessor(input=image)

# create the inference processor
from cvbf.processing.onnx_processor import OnnxProcessor
inference = OnnxProcessor(input=patches, PATH_TO_MODEL_DIR="/app/outer_assets/model/anomaly_detection/")

# create the post-processing processor
from cvbf.postprocessing.patch_processor import PatchPostProcessor
result = PatchPostProcessor(input=inference)

# offload from torch to openCV
from cvbf.offloading.opencv_offloader import OpenCVOffloader
openCVOffloader = OpenCVOffloader(input=result)

# display the image using openCV
from cvbf.offloading.openCVImageShow_offloader import OpenCVImageShowOffloader
openCVImageShowOffloader = OpenCVImageShowOffloader(input=openCVOffloader)

# run the pipeline
inference.load_model()
inference.load_metadata()

image_object.load()

image = image.load(image_object.output[0])

patches.create(384)

inference.run_with_iobinding() # perform processing

result.create_activity_map(cell_size=64, threshold=0.8, resultIndex=0) # perform post-processing

openCVOffloader.convertToOpenCV()

openCVImageShowOffloader.show(height=320, refreshrate=-1) # offload the result by showing the image