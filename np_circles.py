# drawing circles and 100-dimensional hyperspheres with numpy
import numpy as np

space_2d = np.zeros([13, 13])

# we start at the center of the circle, right?
x, y = 6, 6
radius = 4

indices = np.indices(space_2d.shape)
idx_x, idx_y = indices

# distance from center
dist_x = np.absolute(idx_x - x)
dist_y = np.absolute(idx_y - y)

dist_euclid = np.sqrt(dist_x ** 2 + dist_y ** 2)

dist_taxicab = dist_x + dist_y
dist_chebyshev = np.maximum(dist_x, dist_y)

print("euclidean:")
print(dist_euclid.astype(np.int64))

print("taxicab:")
print(dist_taxicab.astype(np.int64))

print("chebyshev:")
print(dist_chebyshev.astype(np.int64))

circle = np.where(dist_euclid < radius)

space_2d[circle] = 1

print(f"circle r = {radius}:")
print(space_2d)

