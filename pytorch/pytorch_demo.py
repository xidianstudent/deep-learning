import torch
import time
from matplotlib import pyplot as plt
import numpy as np

num_inputs = 2
num_examples = 1000
true_w = [2.0, 3.4]
true_b = 2.3
features =  torch.from_numpy(np.random.normal(0.0, 1.0, size=(num_examples, num_inputs)))
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += torch.from_numpy(np.random.normal(0, 0.01,labels.size()))

