# drawing circles and high-dimensional hyperspheres with numpy
import numpy as np
import sys

# never truncate high-d arrays
np.set_printoptions(threshold=sys.maxsize)

dimensionality = 3
dim = np.zeros(dimensionality, dtype=np.int64) + 13 # length-n array of 13s 
space = np.zeros(dim)

# we start at the center of the circle, right?
center = 6 # in any dimension
radius = 4

indices = np.indices(space.shape)

# distance from center
dist = np.absolute(indices - center)

squares = dist ** 2
print(squares)
dist_euclid = np.sqrt(np.sum(squares, axis=0))

dist_taxicab = np.sum(dist, axis=0)
dist_chebyshev = np.maximum.reduce(dist)

print("euclidean:")
print(dist_euclid.astype(np.int64))

print("taxicab:")
print(dist_taxicab.astype(np.int64))

print("chebyshev:")
print(dist_chebyshev.astype(np.int64))

circle = np.where(dist_euclid < radius)

space[circle] = 1

print(f"circle r = {radius}:")
print(space)



