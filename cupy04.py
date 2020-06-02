import numpy as np 
import cupy as cp 
import time

"""
RAW KERNEL
raw kernel allows for the use of raw cuda source
and gives maximum flexibility and customizability
among the 3 kernel types
"""

add_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void my_add(const double* x1, const double* x2, double* y) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        y[tid] = x1[tid] + x2[tid];
    }
    ''', 'my_add')

def add_cpu(x,y):
    return x + y

x1 = np.random.uniform(0,1,100000000)
x1_gpu = cp.asarray(x1)
x2 = np.random.uniform(0,1,100000000)
x2_gpu = cp.asarray(x2)
y = np.zeros(1000)
y_gpu = cp.asarray(y)
#add_kernel((1000,), (0,), (x1_gpu, x2_gpu, y_gpu)) #block size, grid size, arguments
#print(y_gpu)

print("test between numpy and cupy")
print("numpy test")
N = 4
t0 = time.clock()
for i in range(N):
    add_cpu(x1,x2)
t1 = time.clock() - t0
print(t1)
print("cupy test")
t0 = time.clock()
for i in range(N):
    add_kernel((1000,), (0,), (x1_gpu, x2_gpu, y_gpu))
t1 = time.clock() - t0
print(t1)

"""
on HP laptop with Intel i7-7700HQ processor with
NVidia GTX 1050 6GB VRAM with 8GB RAM,
N = 4
the following results were obtained:

test between numpy and cupy
numpy test
5.4434840000000015
cupy test
0.46330799999999783
"""