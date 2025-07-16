import torch
import cv2
from cvbf.opencvprocessing.imageprocessing import ImageProcessing

class DrawPostProcessor(ImageProcessing):
    def __init__(self, input):
        super().__init__(input=input, DEVICE_INDEX=input.DEVICE_INDEX, DEVICE_NAME=input.DEVICE_NAME)

    def add_text(self, text, font=cv2.FONT_HERSHEY_SIMPLEX, bottomLeftCornerOfText=(10,500), fontScale=1, fontColor = (255,255,255), thickness=1, lineType=2) -> None:
        self.output = cv2.putText(self.input.output, text, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
