import torch
import torch.multiprocessing as mp
import time
import onnxruntime
import json
import numpy as np

def load_metadata(PATH_TO_MODEL_DIR):
    return json.load(open(PATH_TO_MODEL_DIR+"/metadata.json"))

def load_model(PATH_TO_MODEL_DIR, DEVICE_INDEX):
    model_path = PATH_TO_MODEL_DIR + "/model.onnx"
    detection_model = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'], provider_options=[{'device_id':str(DEVICE_INDEX)}])
    input_name = detection_model.get_inputs()[0].name
    metadata = json.load(open(PATH_TO_MODEL_DIR+"/metadata.json"))
    return detection_model, input_name, metadata

def inference(queue_input, queue_output, model_path, DEVICE_NAME, DEVICE_INDEX, queue_input_status):
    def inference_batch(detection_model, patches, DEVICE):
        # https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/api/onnxruntime-python-api.py
        binding = detection_model.io_binding()
        patches = patches.contiguous()

        binding.clear_binding_inputs()
        binding.clear_binding_outputs()

        binding.bind_input(
            name=detection_model.get_inputs()[0].name,
            device_type=DEVICE_NAME,
            device_id=DEVICE_INDEX,
            element_type=np.float32,
            shape=tuple(patches.shape),
            buffer_ptr=patches.data_ptr(),
        )

        ## Allocate the PyTorch tensor for the model output
        output = torch.empty((patches.shape[0],1,patches.shape[2],patches.shape[3]), dtype=torch.float32, device=DEVICE).contiguous()
        binding.bind_output(
            name=detection_model.get_outputs()[0].name,
            device_type=DEVICE_NAME,
            device_id=DEVICE_INDEX,
            element_type=np.float32,
            shape=tuple(output.shape),
            buffer_ptr=output.data_ptr(),
        )

        detection_model.run_with_iobinding(binding)
        return output

    DEVICE = f'{DEVICE_NAME}:{DEVICE_INDEX}'

    detection_model, input_name, metadata = load_model(model_path, DEVICE_INDEX)
    queue_input_status.put([1]) # done

    cell_size = 48
    cell_threshold = 0.9

    ones = torch.ones((1,1,1,1)).to(DEVICE)
    zeros = torch.zeros((1,1,1,1)).to(DEVICE)
    pixel_threshold = torch.tensor([metadata['pixel_threshold']]).to(DEVICE)
    threshold = torch.tensor([np.floor((cell_size * cell_size)*cell_threshold)]).to(DEVICE)

    while True:
        print("do something ...")
        time.sleep(1)



class ProcessQueueClass:
    def __init__(self):
        self.qarray = []
        self.qstatus = mp.Queue()
        self.qout = mp.Queue()

    def register_new_queue(self):
        self.qarray.append(mp.Queue())
        print("registered new queue: "+str(len(self.qarray)))
        return len(self.qarray)

    def get_queue_input(self, id):
        return self.qarray[id-1]
    
    def get_queue_output(self):
        return self.qout
    
    def get_number_of_input_queues(self):
        return len(self.qarray)
    
    def get_input_queues(self):
        return [self.qarray]

    def get_input_queues_status(self):
        return self.qstatus


if __name__ == '__main__':
    DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'cpu'

    mp.set_start_method('spawn')

    processing_queues = ProcessQueueClass()

    queue_ids = 0
    processes = []
    queue_ids = processing_queues.register_new_queue()
    processes.append(mp.Process(target=inference, args=(processing_queues.get_queue_input(queue_ids), processing_queues.get_queue_output(), "outer_assets/model/anomaly_detection/", DEVICE_NAME, 0, processing_queues.get_input_queues_status())))
    queue_ids = processing_queues.register_new_queue()
    processes.append(mp.Process(target=inference, args=(processing_queues.get_queue_input(queue_ids), processing_queues.get_queue_output(), "outer_assets/model/anomaly_detection/", DEVICE_NAME, 0, processing_queues.get_input_queues_status())))
    #processes.append(mp.Process(target=prepare, args=(processing_queues.get_input_queues(), processing_queues.get_input_queues_status())))
    #processes.append(mp.Process(target=process, args=(processing_queues.get_queue_output(), queue_ids, DEVICE_NAME)))

    for p in processes:
        p.start()
    for p in processes:
        p.join()
