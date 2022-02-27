import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


a = [True, False, True]
b = [False, True, False]
c = np.array(a) + np.array(b)
print(type(c))
print(c)