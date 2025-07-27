import math
from typing import Union

import numpy as np
import torch

FloatOrTensor = Union[float, torch.Tensor]


# def safe_log(x: FloatOrTensor) -> FloatOrTensor:
#     if torch.is_tensor(x):
#         x[x <= 1e-12] = 0.
#         return torch.log(x)
#     return math.log(x)


# def safe_exp(x: FloatOrTensor) -> FloatOrTensor:
#     if torch.is_tensor(x):
#         return torch.exp(x)
#     return math.exp(x)


def safe_log(x: FloatOrTensor) -> FloatOrTensor:
    if torch.is_tensor(x):
        # Check for very small values
        if torch.any(x <= 1e-12):
            pass
            # print(f"WARNING: safe_log received values ≤ 1e-12: {x[x <= 1e-12]}")
        x_safe = x.clone()
        x_safe[x_safe <= 1e-12] = 1e-12
        return torch.log(x_safe)
    
    if x <= 1e-12:
        # print(f"WARNING: safe_log received small value: {x}")
        return math.log(1e-12)
    return math.log(x)

def safe_exp(x: FloatOrTensor) -> FloatOrTensor:
    #print(f"safe_exp called with x={x}")
    if torch.is_tensor(x):
        # Check for very negative values
        if torch.any(x < -100):
            pass
            # print(f"WARNING: safe_exp received values < -100: {x[x < -100]}")
        return torch.exp(torch.clamp(x, min=-100))
    
    if x < -100:
        # print(f"WARNING: safe_exp received very negative value: {x}")
        return math.exp(-100)
    return math.exp(x)


def safe_log_add(x: FloatOrTensor, y: FloatOrTensor) -> FloatOrTensor:
    if torch.is_tensor(x) or torch.is_tensor(y):
        return torch.logaddexp(torch.as_tensor(x), torch.as_tensor(y))
    return np.logaddexp(x, y)


def safe_log_negate(a: FloatOrTensor, eps=1e-7) -> FloatOrTensor:
    if torch.is_tensor(a):
        return torch.log1p(-torch.exp(a - eps))
    if a > -1e-10:
        return float("-inf")
    return math.log1p(-math.exp(a))
