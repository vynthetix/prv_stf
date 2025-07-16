import torch.multiprocessing as mp

class ProcessQueueClass:
    def __init__(self):
        self.qarray_in = []
        self.qstatus = mp.Queue()
        self.qarray_out = []

    def register_new_queue(self):
        self.qarray_in.append(mp.Queue())
        self.qarray_out.append(mp.Queue())
        print("registered new queue: "+str(len(self.qarray_in)))
        return len(self.qarray_in)

    def get_queue_input(self, id):
        return self.qarray_in[id-1]
    
    def get_queue_output(self, id):
        return self.qarray_out[id-1]
    
    def get_number_of_io_queues(self):
        return len(self.qarray_in)
    
    def get_input_queues(self):
        return [self.qarray_in]

    def get_queues_status(self):
        return self.qstatus

    def get_output_queues(self):
        return [self.qarray_out]