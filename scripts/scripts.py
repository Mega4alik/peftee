import time
import torch
device = torch.device("cuda:0")
t = torch.randn(100,2000,4096).half()
t1 = time.perf_counter()
t = t.to(device, non_blocking=True)
print("t1",  time.perf_counter() - t1)

t1 = time.perf_counter()
t = t.cpu()
print("t2",  time.perf_counter() - t1)
