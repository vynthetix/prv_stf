import torch
import numpy as np
import json
import onnxruntime
from cvbf.processing.processing import Processing
from time import time as now

class OnnxProcessor(Processing):
    def __init__(self, input, PATH_TO_MODEL_DIR) -> None:
        """
        Initialize the OnnxProcessor class.

        Parameters:
        - input (Processing): The input data to be processed.
        - PATH_TO_MODEL_DIR (str): The path to the directory containing the ONNX model and metadata.

        Returns:
        - None
        """
        super().__init__(input=input)
        self.PATH_TO_MODEL_DIR = PATH_TO_MODEL_DIR
        self.metadata = None
        self.dtype = None


    def load_metadata(self) -> None:
        """
        Load the metadata from a JSON file and store it in the 'metadata' attribute.

        Parameters:
        - self (OnnxProcessor): The instance of the OnnxProcessor class.
        - PATH_TO_MODEL_DIR (str): The path to the directory containing the ONNX model and metadata.

        Returns:
        - None

        The function reads the metadata from a JSON file located at 'PATH_TO_MODEL_DIR/metadata.json' and stores it in the 'metadata' attribute of the instance.
        """
        self.metadata = json.load(open(self.PATH_TO_MODEL_DIR+"/metadata.json"))

    def load_model(self) -> None:
        """
        Load the ONNX model from the specified directory and initialize the model attribute.

        Parameters:
        - self (OnnxProcessor): The instance of the OnnxProcessor class.
        - PATH_TO_MODEL_DIR (str): The path to the directory containing the ONNX model.

        Returns:
        - None

        The function constructs the model path by appending "/model.onnx" to the provided PATH_TO_MODEL_DIR.
        Depending on the value of the DEVICE_NAME attribute, it initializes the model attribute using either the CUDAExecutionProvider or the CPUExecutionProvider.
        If DEVICE_NAME is 'cuda', it sets the device_id using the DEVICE_INDEX attribute.
        """
        model_path = self.PATH_TO_MODEL_DIR + "/model.onnx"
        if self.DEVICE_NAME == 'cuda':
            self.model = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'], provider_options=[{'device_id':str(self.DEVICE_INDEX)}])
        elif self.DEVICE_NAME == 'cpu':
            self.model = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])


    def __create_batches(self, max_batch_size: int = None) -> torch.Tensor:
        """
        This function creates batches from the input tensor for efficient processing.

        Parameters:
        - max_batch_size (int, optional): The maximum number of tensors in a single batch. If not provided, the function will return a single batch containing all tensors.

        Returns:
        - torch.Tensor: A tensor containing the batches of input tensors. If max_batch_size is provided and is less than the number of tensors in the input, the function will return a tensor with multiple batches. Otherwise, it will return a single batch containing all tensors.
        """
        batches = self.prepareInput(self.input.output[0])
        if max_batch_size is not None and max_batch_size < batches.shape[0]:
            return torch.split(tensor=batches, split_size_or_sections=max_batch_size, dim=0)
        else:
            return batches.unsqueeze(0)

    def prepareInput(self, batch) -> None:
        if self.model.get_inputs()[0].type == "tensor(uint8)":
            self.dtype = np.uint8
            batch = (batch*255).type(torch.uint8)
        elif self.model.get_inputs()[0].type:
            self.dtype = np.float32
            batch = (batch).type(torch.float)
        else:
            print("cannot prepare input tensor of type: "+self.model.get_inputs()[0].type)
            batch = None
        
        channel_index = [i for i, x in enumerate(self.model.get_inputs()[0].shape) if x == 3][0]
        if channel_index == 1:
            batch = batch
        elif channel_index == 3:
            batch = batch.permute(0,2,3,1)
        else:
            print("cannot find where to put the channel dimension.")
            batch = None
        return batch

    def run(self, max_batch_size: int = None) -> None:
        """
        Run the ONNX model on the input data, creating batches if specified.

        Parameters:
        - max_batch_size (int, optional): The maximum number of tensors in a single batch. If not provided, the function will process all tensors as a single batch.

        Returns:
        - None

        The function performs the following steps:
        1. Calls the `__create_batches` method to create batches of input tensors based on the provided `max_batch_size`.
        2. Iterates over each batch, runs the ONNX model using the input tensor, and appends the result to the `result` list.
        3. Concatenates all the results in the `result` list along the first dimension to form the final output tensor, which is stored in the `output` attribute.
        """
        start = now()
        result = []
        for batch in self.__create_batches(max_batch_size):
            res = []
            for r in self.model.run(None, {self.model.get_inputs()[0].name: batch.cpu().numpy().astype(self.dtype)}):
                res.append(torch.from_numpy(r).to(f'{self.DEVICE_NAME}:{self.DEVICE_INDEX}'))
        
            result.append(res)

        self.rearrangeOutput(result, start)

        

    def run_with_iobinding(self, max_batch_size: int = None) -> None:
        """
        Run the ONNX model on the input data using I/O binding, creating batches if specified.

        Parameters:
        - max_batch_size (int, optional): The maximum number of tensors in a single batch. If not provided, the function will process all tensors as a single batch.

        Returns:
        - None

        The function performs the following steps:
        1. Calls the `__create_batches` method to create batches of input tensors based on the provided `max_batch_size`.
        2. Iterates over each batch, runs the ONNX model using the input tensor and I/O binding, and appends the result to the `result` list.
        3. Concatenates all the results in the `result` list along the first dimension to form the final output tensor, which is stored in the `output` attribute.
        """
        start = now()
        result = []
        for batch in self.__create_batches(max_batch_size):
            # https://github.com/microsoft/onnxruntime-inference-examples/blob/main/python/api/onnxruntime-python-api.py
            binding = self.model.io_binding()
            binding.clear_binding_inputs()
            binding.clear_binding_outputs()

            batch = batch.contiguous()
            binding.bind_input(
                name=self.model.get_inputs()[0].name,
                device_type=self.DEVICE_NAME,
                device_id=self.DEVICE_INDEX,
                element_type=self.dtype,
                shape=tuple(batch.shape),
                buffer_ptr=batch.data_ptr(),
            )

            ## Allocate the PyTorch tensor for the model output
            res = []
            for o in self.model.get_outputs():
                if "outputs" not in self.metadata:
                    # use the information directly provided by the model
                    output = [out for out in self.metadata['outputs'] if out['name'] == o.name][0]
                    outputName = output.name
                    outputShape = o.shape
                    res.append(torch.empty(size=outputShape, dtype=torch.float, device=self.DEVICE).contiguous())
                else:
                    # try to match information possibly provided in the metadata
                    outputName = o.name
                    outputShape = (batch.shape[0],1,batch.shape[2],batch.shape[3])
                    res.append(torch.empty(size=outputShape, dtype=torch.float, device=self.DEVICE).contiguous())

                binding.bind_output(
                    name=outputName,
                    device_type=self.DEVICE_NAME,
                    device_id=self.DEVICE_INDEX,
                    element_type=np.float32,
                    shape=outputShape,
                    buffer_ptr=res[len(res)-1].data_ptr(),
                )

            self.model.run_with_iobinding(binding)
            result.append(res)

        self.rearrangeOutput(result, start)

    def rearrangeOutput(self, result, start):
        # rearrange the output to match the form [nbr_outputs,num_images,...]
        self.output = []
        for i in range(0, len(result[0])):
            self.output.append(torch.cat([r[i] for r in result], 0))
        self.output = [self.output, now()-start]
            