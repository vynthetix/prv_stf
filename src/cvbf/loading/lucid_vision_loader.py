from cvbf.loading.loading import Loading
import torch
import cvbf
#import gi
#gi.require_version("Aravis", "0.8") # or whatever version number you have installed
#from gi.repository import Aravis


# setup aravis plugin
class UserData:
    def __init__(self) -> None:
        self.stream = None
        self.frame = None
        self.frame_id = None
        self.timestamp = None
        self.locked = False

    def set_frame(self, frame):
        self.frame = frame

    def get_frame(self):
        return self.frame

    def set_timestamp(self, timestamp):
        self.timestamp = time.time()

    def get_timestamp(self):
        return self.timestamp

    def set_frame_id(self, frame_id):
        self.frame_id = frame_id

    def get_frame_id(self):
        return self.frame_id

    def get_data(self):
        if self.locked:
            return None, None, None
        else:
            return self.frame, self.frame_id, self.timestamp

    def lock(self):
        self.locked = True

    def unlock(self):
        self.locked = False


class LucidVision(Loading):
    def __init__(self, DEVICE_NAME, DEVICE_INDEX) -> None:
        super().__init__(DEVICE_NAME=DEVICE_NAME, DEVICE_INDEX=DEVICE_INDEX)
        self.cam = Aravis.Camera.new()
        self.Aravis = Aravis
        self.stream = None

    def callback(self, user_data, cb_type, buffer):
        if buffer is not None:
            self.lock(self.id)
            self.set_data(self.convert(buffer)[[2,1,0],...], buffer.get_frame_id(), buffer.get_timestamp(), self.id)
            self.unlock(self.id)
            self.stream.push_buffer(buffer)

    def convert(self, buf):
        #https://github.com/AravisProject/aravis/issues/453
        if not buf:
            return None
        try:
            return (torch.frombuffer(buf.get_image_data(), dtype=torch.uint8).to(self.DEVICE)/255).reshape(buf.get_image_height(), buf.get_image_width(), 3).flip(2).swapaxes(2,1).swapaxes(1,0)
        except ValueError:
            return None

    def run(self, ):
        user_data = UserData()
        self.stream = self.cam.create_stream(self.callback, user_data)
        user_data.stream = self.stream
        payload = self.cam.get_payload()
        self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))
        self.cam.start_acquisition()