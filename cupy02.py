import numpy as np 
import cupy as cp 
import time

"""
In computing, a compute kernel 
is a routine compiled for high throughput accelerators 
(such as graphics processing units (GPUs), 
digital signal processors (DSPs) or 
field-programmable gate arrays (FPGAs)), 
separate from but used by a main program 
(typically running on a central processing unit).
"""

"""
3 types of kernels:
elementwise,
reduction,
raw
"""

"""
ELEMENTWISE KERNELS
A definition of an elementwise kernel
consists of four parts:
an input argument list,
an output argument list,
a loop body code,
and the kernel name. 
For example, a kernel that computes
a squared difference f(x,y)=(x−y)^2
is defined as follows:
"""

x = np.random.uniform(0,5,1000000)
x_gpu = cp.asarray(x)
y = np.random.uniform(0,5,1000000)
y_gpu = cp.asarray(y)

def squared_diff_cpu(x,y):
    z = np.zeros(len(x))
    for i in range(len(x)):
        z[i] = x[i]*x[i] - y[i]*y[i]
    return z

squared_diff_gpu = cp.ElementwiseKernel(
    'float64 x, float64 y',
    'float64 z',
    'z = (x*x - y*y)',
    'squared_diff_gpu')
N = 4
print("test between numpy and cupy")
print("numpy test")
t0 = time.clock()
for i in range(N):
    squared_diff_cpu(x,y)
t1 = time.clock() - t0
print(t1)
print("cupy test")
t0 = time.clock()
for i in range(N):
    squared_diff_gpu(x_gpu,y_gpu)
t1 = time.clock() - t0
print(t1)

"""
on HP laptop with Intel i7-7700HQ processor with
NVidia GTX 1050 6GB VRAM with 8GB RAM,
N = 4
the following results were obtained:

test between numpy and cupy
numpy test
12.946543
cupy test
0.3895710000000001
"""