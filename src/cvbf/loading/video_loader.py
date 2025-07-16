import torch
import cv2
import time
from threading import Thread
from cvbf.loading.loading import Loading

class VideoLoader(Loading):
    def __init__(self, DEVICE_NAME, DEVICE_INDEX, PATH_TO_VIDEO, fps) -> None:
        """
        Initialize a VideoLoader object.

        Parameters:
        - DEVICE_NAME (str): The name of the device to use for processing.
        - DEVICE_INDEX (int): The index of the device to use for processing.
        - PATH_TO_VIDEO (str): The path to the video file to load.

        Returns:
        - None
        """
        super().__init__(DEVICE_NAME=DEVICE_NAME, DEVICE_INDEX=DEVICE_INDEX)
        self.fps = fps
        self.frame_id = 0
        self.isPlaying = False
        self.__load(PATH_TO_VIDEO=PATH_TO_VIDEO)


    def __load(self, PATH_TO_VIDEO) -> None:
        """
        Load a video file into a VideoCapture object.

        This method initializes a VideoCapture object using the provided video file path.
        It also sets the frame sleep time to 1/1.0 seconds.

        Parameters:
        - PATH_TO_VIDEO (str): The path to the video file to load.

        Returns:
        - None
        """
        self.__videoReader = cv2.VideoCapture(PATH_TO_VIDEO)
        self.__frame_sleep = 1.0/self.fps


    def _play(self):
        """
        Private method to play the video in a separate thread.

        This method reads frames from the video file, processes them, and sends them to the processing pipeline.
        It also handles synchronization with the processing pipeline using the lock mechanism.

        Parameters:
        - None

        Returns:
        - None
        """
        if self.__videoReader.isOpened():
            self.isPlaying = True
            ret, frame = self.__videoReader.read()
            while ret and self.isPlaying:
                image = torch.from_numpy(frame[:,:,[2,1,0]]/255.0).to(self.DEVICE).permute(2,0,1)
                while self.lock(self.id) == False:
                    time.sleep(0.01)
                if len(image.shape) < 3:
                    image = image.unsqueeze(0).repeat(3,1,1)
                self.set_data(image,
                            self.frame_id,
                            time.time(),
                            self.id)
                self.unlock(self.id)
                self.frame_id = self.frame_id + 1
                ret, frame = self.__videoReader.read()
                time.sleep(self.__frame_sleep)
            self.isPlaying = False

    def play(self) -> None:
        """
        Start playing the video in a separate thread.

        This method creates a new thread and starts playing the video by calling the private method `_play`.
        The video is read frame by frame, processed, and sent to the processing pipeline.
        Synchronization with the processing pipeline is handled using the lock mechanism.

        Parameters:
        - None

        Returns:
        - None
        """
        Thread(target=self._play).start()

    def next(self):
        """
        Updates the input data and prepares the output.

        The method locks the input object, retrieves the latest data, and
        unlocks the input object. The retrieved data is then stored in the
        'frame', 'frame_id', and 'frame_timestamp' attributes.
        """
        self.lock(self.id)
        self.output = self.get_data(self.id)
        self.unlock(self.id)