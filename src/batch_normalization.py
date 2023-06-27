import numpy as np


def batch_normalization(x, gamma, beta, moving_mean, moving_var, eps, momentum, train=True):
    # if not using_config('train'):
    if not train:
        x_normed = (x - moving_mean) / np.sqrt(moving_var + eps)
    else:
        x_mean = np.mean(x, axis=0)
        x_var = np.var(x, axis=0)
        x_normed = (x - x_mean) / np.sqrt(x_var + eps)
        moving_mean *= momentum
        moving_mean += (1 - momentum) * x_mean
        moving_var *= momentum
        moving_var += (1 - momentum) * x_var
    out = gamma * x_normed + beta
    return out
