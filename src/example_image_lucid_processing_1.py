import torch
import time

DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE_INDEX = 0

from cvbf.loading.lucid_vision_loader import LucidVision
lucidVision = LucidVision(DEVICE_NAME=DEVICE_NAME, DEVICE_INDEX=DEVICE_INDEX)

from cvbf.loading.image_object_loader import ImageObjectLoader
imageObjectLoader = ImageObjectLoader(DEVICE_NAME=DEVICE_NAME, DEVICE_INDEX=DEVICE_INDEX)

from cvbf.offloading.opencv_offloader import OpenCVOffloader
openCVOffloader = OpenCVOffloader(input=imageObjectLoader)

# display the image using openCV
from cvbf.offloading.openCVImageShow_offloader import OpenCVImageShowOffloader
openCVImageShowOffloader = OpenCVImageShowOffloader(input=openCVOffloader)


lucidVision.cam.set_frame_rate(1.0)
lucidVision.cam.set_pixel_format(lucidVision.Aravis.PIXEL_FORMAT_RGB_8_PACKED)
lucidVision.cam.set_integer("TargetBrightness", 50)
lucidVision.cam.set_acquisition_mode(lucidVision.Aravis.AcquisitionMode.CONTINUOUS)
lucidVision.cam.set_string("TriggerSelector", "FrameStart")
lucidVision.cam.set_string("TriggerSource", "Line0")
lucidVision.cam.set_string("TriggerActivation", "LevelLow")
lucidVision.cam.set_string("TriggerOverlap", "Off")
lucidVision.cam.set_string("TriggerLatency", "Off")
lucidVision.cam.set_string("TriggerMode", "Off")



lucidVision.run()
old_frame_id = -1
while True:
    imageInfo = lucidVision.output
    if imageInfo[1] is not None:
        if imageInfo[1] != old_frame_id:
            imageObjectLoader.load(image=imageInfo[0])
            openCVOffloader.convertToOpenCV()
            openCVImageShowOffloader.show(height=320, refreshrate=1)
            old_frame_id = imageInfo[1]
    time.sleep(0.01)

