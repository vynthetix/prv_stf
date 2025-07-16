import time
import torch
import cvbf

DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE_INDEX = 0

# load a sample file
from cvbf.loading.video_loader import VideoLoader
video = VideoLoader(DEVICE_NAME=DEVICE_NAME, DEVICE_INDEX=DEVICE_INDEX, PATH_TO_VIDEO="outer_assets/samples/01.mp4", fps=6.0)

# create the pre-processing object
from cvbf.preprocessing.patch_preprocessor import PatchPreProcessor
patches = PatchPreProcessor(input=video)

# create the inference object
from cvbf.processing.onnx_processor import OnnxProcessor
inference = OnnxProcessor(input=patches, PATH_TO_MODEL_DIR="outer_assets/model/anomaly_detection/")

from cvbf.postprocessing.patch_processor import PatchPostProcessor
result = PatchPostProcessor(input=inference)

from cvbf.postprocessing.imageOverlay_postprocessor import ImageOverlayPostProcessor
overlay = ImageOverlayPostProcessor(input=result)

# offload from torch to openCV
from cvbf.offloading.opencv_offloader import OpenCVOffloader
openCVOffloader = OpenCVOffloader(input=overlay)

# display the image using openCV
from cvbf.offloading.openCVImageShow_offloader import OpenCVImageShowOffloader
openCVImageShowOffloader = OpenCVImageShowOffloader(input=openCVOffloader)

# processing
inference.load_model()
inference.load_metadata()
video.play() # start the video
old_frame_id = None
while True:
    video.next()
    if old_frame_id != video.output[1]:
        print("processing frame "+str(video.output[1])+" on device: "+str(video.DEVICE))
        video.output[0] = video.output[0][:,0:(3*384),0:(10*384)]
        patches.create(384)
        inference.run(max_batch_size=10)
        result.create_activity_map(cell_size=64, threshold=0.8, resultIndex=0)
        overlay.underlay(underlay=video.output[0], alpha=0.5)
        openCVOffloader.convertToOpenCV()
        openCVImageShowOffloader.show(height=320, refreshrate=10)
        old_frame_id = video.output[1]
    else:
        time.sleep(0.01)
