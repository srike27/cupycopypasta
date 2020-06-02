import numpy as np 
import cupy as cp 
import time

"""
REDUCTION KERNEL
when processing the kernel into a simpler unit
Identity value: This value is used for the initial value of reduction.
Mapping expression: It is used for the pre-processing of each element to be reduced.
Reduction expression: It is an operator to reduce the multiple mapped values. The special variables a and b are used for its operands.
Post mapping expression: It is used to transform the resulting reduced values. The special variable a is used as its input. Output should be written to the output parameter.
"""

normalize_gpu = cp.ReductionKernel(
    'float64 x', #input params
    'float64 y', #output params
    'x * x', # map
    'a + b', # reduce
    'y = sqrt(a)', # post reduction map
    '0', # identity value
    'normalize_gpu' # name
)


x = cp.array([0.0,6.0,8.0])
print(normalize_gpu(x))
