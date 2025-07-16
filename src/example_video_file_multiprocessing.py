import torch
import torch.multiprocessing as mp
import time
import uuid
import pprint
from cvbf.multiprocessing.PipelineClass import PipelineClass
from cvbf.multiprocessing.ProcessQueueClass import ProcessQueueClass
from cvbf.multiprocessing.QueueObjectClass import QueueObjectClass

class P(pprint.PrettyPrinter):
  def _format(self, object, *args, **kwargs):
    if isinstance(object, str):
      if len(object) > 20:
        object = object[:20] + '...'
    elif isinstance(object, torch.Tensor):
        if len(object.shape) > 1:
            object = object.shape
        else:
            object = str(object)
            if len(object) > 20:
                object = object[:20] + '...'
    return pprint.PrettyPrinter._format(self, object, *args, **kwargs)

class Input():
    def __init__(self, input, input_queues: list, DEVICE_NAME: str, DEVICE_INDEX: int) -> None:
        # load the file
        self.PATH_TO_VIDEO = input
        self.DEVICE_INDEX = DEVICE_INDEX
        self.DEVICE_NAME = DEVICE_NAME
        self.input_queues = input_queues
        self.id = uuid.uuid1()

        from cvbf.loading.video_loader import VideoLoader
        self.input = VideoLoader(DEVICE_INDEX=self.DEVICE_INDEX, DEVICE_NAME=self.DEVICE_NAME, PATH_TO_VIDEO=self.PATH_TO_VIDEO, fps=6.0)

    def run(self) -> None:
        self.input.play()
        counter = 0
        frame_id_old = -1
        while True:
            self.input.lock(self.id)
            frame, frame_id, timestamp = self.input.get_data(self.id)
            self.input.unlock(self.id)
            if frame != None and frame_id != frame_id_old:
                image = frame[:,0:(3*384),0:(10*384)]
                C, H, W = image.shape
                selection = torch.arange(0, int((W/384 * H/384)/len(self.input_queues[0])))
                queues_empty = True
                for i in range(0, len(self.input_queues[0])):
                    if self.input_queues[0][i].qsize() > 0:
                        queues_empty = False
                        break
                if queues_empty == True:
                    for i in range(0, len(self.input_queues[0])):
                        object = QueueObjectClass(image=image.cpu().numpy(),
                                                                        metadata={'id': self.input.id,
                                                                                'frame_id': frame_id,
                                                                                'queue_id': i,
                                                                                'patch_selection': (selection+((torch.max(selection)+1)*i).to(self.input.DEVICE))})
                        self.input_queues[0][i].put(object)
                    counter = counter + 1
                    frame_id_old = frame_id
            time.sleep(0.01)

class Pipeline(PipelineClass):
    def __init__(self, input_queue, output_queue, DEVICE_NAME, DEVICE_INDEX) -> None:
        super().__init__(input_queue, output_queue, DEVICE_NAME, DEVICE_INDEX)

        from cvbf.loading.image_object_loader import ImageObjectLoader
        self.image = ImageObjectLoader(DEVICE_NAME=self.DEVICE_NAME, DEVICE_INDEX=self.DEVICE_INDEX)

        #  create the pre-processing processor
        from cvbf.preprocessing.patch_preprocessor import PatchPreProcessor
        self.patches = PatchPreProcessor(input=self.image)

        # create the inference processor
        from cvbf.processing.onnx_processor import OnnxProcessor
        self.inference = OnnxProcessor(input=self.patches, PATH_TO_MODEL_DIR="outer_assets/model/anomaly_detection/")
    
    def run(self):
        while True:
            if self.input_queue.qsize() > 0:
                start = time.time()
                input = self.input_queue.get()
                self.image.load(torch.from_numpy(input.get_image()).to(self.inference.DEVICE))
                self.patches.create(384)
                self.patches.select(indices=input.get_metadata()["patch_selection"])
                input.get_metadata()['nbr_rows'] = self.patches.nbr_rows
                input.get_metadata()['nbr_cols'] = self.patches.nbr_cols
                input.get_metadata()['image'] = self.patches.output
                input.get_metadata()['runtime_preprocessing'] = round((time.time()-start)*1000, 4)
                start = time.time()
                self.inference.run_with_iobinding(max_batch_size=3)
                input.get_metadata()['runtime_inference'] = round((time.time()-start)*1000, 4)
                self.output_queue.put(QueueObjectClass(image=self.inference.output.to(self.image.DEVICE), metadata=input.get_metadata()))
            else:
                time.sleep(0.01)

class Output():
    def __init__(self, output_queues, DEVICE_NAME, DEVICE_INDEX) -> None:
        self.output_queues = output_queues
        self.DEVICE_NAME = DEVICE_NAME
        self.DEVICE_INDEX = DEVICE_INDEX

        from cvbf.loading.image_object_loader import ImageObjectLoader
        self.image = ImageObjectLoader(DEVICE_NAME=self.DEVICE_NAME, DEVICE_INDEX=self.DEVICE_INDEX)

        #  create the dummy pre-processing processor
        from cvbf.preprocessing.patch_preprocessor import PatchPreProcessor
        self.patches = PatchPreProcessor(input=self.image)

        # create the dummy inference processor
        from cvbf.processing.onnx_processor import OnnxProcessor
        self.inference = OnnxProcessor(input=self.patches, PATH_TO_MODEL_DIR="outer_assets/model/anomaly_detection/")
        self.inference.load_metadata()

        # create the post-processing processor
        from cvbf.postprocessing.patch_processor import PatchPostProcessor
        self.result = PatchPostProcessor(input=self.inference)

        from cvbf.postprocessing.imageOverlay_postprocessor import ImageOverlayPostProcessor
        self.overlay = ImageOverlayPostProcessor(input=self.result)

        # offload from torch to openCV
        from cvbf.offloading.opencv_offloader import OpenCVOffloader
        self.openCVOffloader = OpenCVOffloader(input=self.overlay)

        # display the image using openCV
        from cvbf.offloading.openCVImageShow_offloader import OpenCVImageShowOffloader
        self.openCVImageShowOffloader = OpenCVImageShowOffloader(input=self.openCVOffloader)

    def run(self) -> None:
        while True:
            tensors = []
            while len(tensors) < len(self.output_queues[0]):
                for i in range(0, len(self.output_queues[0])):
                    while self.output_queues[0][i].qsize() == 0:
                        time.sleep(0.01)
                    tensors.append(self.output_queues[0][i].get())
            tensors.sort(key=lambda x: x.get_metadata()["queue_id"], reverse=False)
            print("collected all tensors for frame with id: "+str(tensors[0].get_metadata()["frame_id"]))
            nbr_rows = None
            nbr_cols = None
            for tensor in tensors:
                nbr_rows = tensor.get_metadata()["nbr_rows"]
                nbr_cols = tensor.get_metadata()["nbr_cols"]
                P().pprint(tensor.get_metadata())
            self.patches.output = torch.cat([tensor.get_image() for tensor in tensors], 0)
            self.patches.nbr_cols = nbr_cols
            self.patches.nbr_rows = nbr_rows
            self.inference.output = self.inference.input.output
            self.result.create_activity_map(cell_size=64, threshold=0.8, resultIndex=0)

            images = torch.cat([tensor.get_metadata()["image"] for tensor in tensors], 0)
            image = self.result.stitch_patches(patches=images, nbr_cols=nbr_cols, nbr_rows=nbr_rows)
            self.overlay.underlay(underlay=image[0], alpha=0.2)
            self.openCVOffloader.convertToOpenCV()
            self.openCVImageShowOffloader.show(height=320)

def create_pipeline(input_queue, output_queue, DEVICE_NAME, DEVICE_INDEX) -> None:
        pipeline = Pipeline(input_queue, output_queue, DEVICE_NAME, DEVICE_INDEX)
        pipeline.inference.load_model()
        pipeline.inference.load_metadata()
        pipeline.run()

def create_input(input, input_queues, DEVICE_NAME, DEVICE_INDEX) -> None:
    input = Input(input, input_queues, DEVICE_NAME, DEVICE_INDEX)
    input.run()

def create_output(output_queues, DEVICE_NAME, DEVICE_INDEX) -> None:
    output = Output(output_queues, DEVICE_NAME, DEVICE_INDEX)
    output.run()

if __name__ == '__main__':
    DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE_INDEX = 0

    mp.set_start_method('spawn')

    processes = []
    pipelines = []

    processing_queues = ProcessQueueClass()

    # start the pipeline processes
    nbr_processes = 2
    for i in range(0, nbr_processes):
        queue_ids = processing_queues.register_new_queue()
        processes.append(mp.Process(target=create_pipeline, args=(processing_queues.get_queue_input(id=queue_ids), processing_queues.get_queue_output(id=queue_ids), DEVICE_NAME, 0)))
    processes.append(mp.Process(target=create_input, args=("outer_assets/samples/01.mp4", processing_queues.get_input_queues(), 'cpu', 0)))
    processes.append(mp.Process(target=create_output, args=(processing_queues.get_output_queues(), DEVICE_NAME, DEVICE_INDEX)))

    for p in processes:
        p.start()
    for p in processes:
        p.join()
