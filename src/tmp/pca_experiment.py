import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

print("===========================")
x = np.array([
    [random.randint(0, 100) for _ in range(100)] for _ in range(100)
])
pca = PCA(n_components=10)
pca.fit(x)
x_pca = pca.transform(x)
print(x.shape, x_pca.shape)

print("===========================")

x_new = np.array([
    [random.randint(0, 100) for _ in range(100)] for _ in range(1)
])
x_new_pca = pca.transform(x_new)
print(x_new.shape, x_new_pca.shape)
