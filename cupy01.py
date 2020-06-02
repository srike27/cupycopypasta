import numpy as np 
import cupy as cp 
import time 

x_gpu = cp.array([1,0,1,1]) #declaring a cupy array
norm = cp.linalg.norm(x_gpu) #test inbuilt function that works on gpu
#print(norm)

# cp.cuda.Device(n).use() for swapping gpu on multi gpu systems

"""
Operations are performed on current device
therefore once gpu switched operations on variable
declared with previous device must not be run.
"""

x = np.array([1,2,3])
x2_gpu = cp.asarray(x) # this function can be used to pass np arrays to cp arrays
x3 = cp.asnumpy(x2_gpu) # this function can be used to pass cp arrays to numpy arrays

"""
For cpu/gpu agnostic code this can be used:
"""

#xp = cp.get_array_module(x)
N = 1000
N2 = 1000000
y = np.random.uniform(0,1,N2)
y_gpu = cp.asarray(y)

print("test between numpy and cupy")
print("numpy test")
t0 = time.clock()
for i in range(N):
    np.linalg.norm(y)
t1 = time.clock() - t0
print(t1)
print("cupy test")
t0 = time.clock()
for i in range(N):
    cp.linalg.norm(y_gpu)
t1 = time.clock() - t0
print(t1)

"""
on HP laptop with Intel i7-7700HQ processor with
NVidia GTX 1050 6GB VRAM with 8GB RAM,
N = 1000, N2 = 1000000
the following results were obtained:

test between numpy and cupy
numpy test
7.684543
cupy test
1.631170000000001
"""