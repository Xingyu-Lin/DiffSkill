import torch
import torch as th
import numpy as np

DEFAULT_DTYPE = torch.float32
device = 'cuda'

def set_default_tensor_type(dtypecls):
    global DEFAULT_DTYPE
    th.set_default_tensor_type(dtypecls)
    if dtypecls is th.DoubleTensor:
        DEFAULT_DTYPE = torch.float64
    else:
        DEFAULT_DTYPE = torch.float32


def np2th(nparr):
    dtype = DEFAULT_DTYPE if nparr.dtype in (np.float64, np.float32) else None
    return th.from_numpy(nparr).to(device=device, dtype=dtype)


def np2th_cpu(nparr):
    dtype = DEFAULT_DTYPE if nparr.dtype in (np.float64, np.float32) else None
    return th.from_numpy(nparr).to(dtype=dtype)
