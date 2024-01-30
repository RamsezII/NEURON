import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

print("PyTorch version: ", torch.__version__)
print("Numpy version: ", np.__version__)
# print("Matplotlib version: ", plt.__version__)
print("GPU available: ", torch.cuda.is_available())
print("CuDNN enabled: ", torch.backends.cudnn.enabled)
print("CuDNN version: ", torch.backends.cudnn.version())
print("Number of CPU threads: ", torch.get_num_threads())
print("Number of GPU: ", torch.cuda.device_count())
print("Current GPU: ", torch.cuda.current_device())
print("Name of current GPU: ", torch.cuda.get_device_name(torch.cuda.current_device()))
print("Memory cached stats: ", torch.cuda.memory_cached(torch.cuda.current_device()))
print("Memory allocated stats: ", torch.cuda.memory_allocated(torch.cuda.current_device()))

