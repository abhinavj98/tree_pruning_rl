import random
import torch
import numpy as np


# repro.
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print('torch version: {}'.format(torch.__version__))

x = torch.randn(1, 3, 10, 10)
x[0, 0, 0, 0] = torch.log(torch.tensor([-1.]))

m = torch.nn.Conv2d(3, 6, 3, 1, 1)
output = m(x)
print(output[0, 0, 0:3, 0:3])

output = m(x)
print(output[0, 0, 0:3, 0:3])

r = torch.nn.ReLU()
output = r(output)
print(output[0, 0, 0:3, 0:3])