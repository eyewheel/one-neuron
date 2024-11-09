# calculate how high-dimensional the hypercubes can get
# don't allocate 100-dimensional tensors at home, kids

import numpy as np

# 13 is just a good size at which the circles appear circular
# when taking a 2d slice of the sphere.

i = 1
while True:
   shape = np.zeros(i, dtype=np.int8) + 13
   dim_space = np.zeros(shape, dtype=np.int8)
   siz = dim_space.size * dim_space.itemsize
   units = ['B', 'KB', 'MB', 'GB']
   unit_idx = 0
   while siz >= 1024:
       siz /= 1024
       unit_idx += 1

   print(f"{i}-dimensional space is {np.trunc(siz)}{units[unit_idx]}")
   i += 1
   input(">")

# terminates at about 13^9

# max memsize on 64-bit is 2^64
# 13 ~= 2^3
# 13^9 ~= 2^3^9 ~= 2^27
# so not theoretical max memsize but some other issue (memory fragmentation probably?)

