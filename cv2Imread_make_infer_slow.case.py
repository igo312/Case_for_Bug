import time
import numpy as np
import cv2


import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


# Return a batch of image dir  when `send` is called
class Team:
    def __init__(self):
       pass

        
    def run(self, callback):
        for _ in range(400):
            callback()
       

team = Team()

class RUNNER(object):
    def __init__(self, engine, batch_size):
        #cuda.init()

        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        trt.init_libnvinfer_plugins(logger,'')
        
        self.batch_size = batch_size
        self.context = engine.create_execution_context()
        self.imgsz = engine.get_binding_shape(0)[2:]
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            if engine.binding_is_input(binding):
                self.inp_size = size 
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(device_mem))
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                self.bindings.append(int(device_mem))
                self.outputs.append({'host': host_mem, 'device': device_mem})



    def _infer(self, img):
        
        infer_num = img.shape[0]
        # padding img if the last is less than batch_size 
        img_flatten = np.ravel(img)
        pad_zeros = np.zeros(self.inp_size - img_flatten.shape[0], dtype=np.float32)
        img_inp = np.concatenate([img_flatten, pad_zeros], axis=0)
        self.inputs[0]['host'] = img_inp

        for inp in self.inputs:
            cuda.memcpy_htod(inp['device'], inp['host'])

        # run inference
        self.context.execute_v2(
            bindings=self.bindings,
            )

        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'])

        # synchronize stream
        data = [out['host'] for out in self.outputs]
        return infer_num, data


def _get_engine(engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    logger.min_severity = trt.Logger.Severity.ERROR
    runtime = trt.Runtime(logger)
    trt.init_libnvinfer_plugins(logger,'') # initialize TensorRT plugins
    with open(engine_path, "rb") as f:
        serialized_engine = f.read()
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine

batch_size = 1
engine_path = './test.batch1.fp16.trt'
engine = _get_engine(engine_path)
runner = RUNNER(engine, batch_size)

pad_img = np.load('./data_640x384_batch1_nonorm.npy')

def my_callback():
    cv2.imread("./00041.jpg")
    astart = time.time()
    runner._infer(pad_img)
    aend = time.time()
    
    runner_time = aend - astart
    print("runner time is %.3f"%(runner_time))
    return 

team.run(my_callback)
