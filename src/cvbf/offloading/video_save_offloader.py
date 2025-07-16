import torchvision
import torch
from cvbf.offloading.offload import Offload

class VideoSaveOffloader(Offload):
    """
    Offloads video frames to a buffer and writes them to a video file when the buffer is full.

    Parameters:
    - input (object): The input source providing video frames.
    - VIDEO_FILE_PATH (str): The file path where the video will be saved.
    - fps (int): The frames per second for the output video.
    - video_buffer (int): The maximum number of frames to buffer before writing to the video file.

    Attributes:
    - video (torch.Tensor): A tensor to store the buffered video frames.
    - fps (int): The frames per second for the output video.
    - index (int): The current index in the video buffer.
    - video_buffer (int): The maximum number of frames to buffer before writing to the video file.

    Methods:
    - add(): Adds a frame to the video buffer.
    - write(): Writes the buffered video frames to a video file.
    """

    def __init__(self, input, VIDEO_FILE_PATH, fps, video_buffer) -> None:
        super().__init__(input=input)
        self.VIDEO_FILE_PATH = VIDEO_FILE_PATH
        self.video = None
        self.fps = fps
        self.index = 0
        self.video_buffer = video_buffer


    def add(self, ):
        frame = self.input.output
        if frame.shape[2] == 1:
            frame = frame.expand(1, 3, frame.shape[1], frame.shape[2])

        if self.video is None:
            self.video = torch.zeros((self.video_buffer, frame.shape[1], frame.shape[2], frame.shape[3])).to(self.DEVICE)
            self.video[0,:] = frame[0]
            self.index = self.index + 1
        else:
            if self.index < self.video_buffer:
                self.video[self.index,:] = frame[0,:]
                self.index = self.index + 1
            else:
                return False
        return True

    
    def write(self, ):
        print("writing "+str(self.index)+" images to video file ...")
        torchvision.io.write_video(filename=self.VIDEO_FILE_PATH, video_array=(self.video.permute(0,3,2,1)[0:self.index]*255).cpu().type(torch.uint8).numpy(), options = {"crf": "17"}, fps=self.fps)


