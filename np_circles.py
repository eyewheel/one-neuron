# drawing circles and 100-dimensional hyperspheres with numpy
import numpy as np

space_2d = np.zeros([13, 13])

# we start at the center of the circle, right?
x, y = 6, 6
radius = 4

indices = np.indices(space_2d.shape)
idx_x, idx_y = indices

# distance from center
dist_x = idx_x - x
dist_y = idx_y - y

index_mask = np.sqrt(dist_x ** 2 + dist_y ** 2)

print(index_mask.astype(np.int64))

circle = np.where(index_mask < radius)

space_2d[circle] = 1

print(space_2d)

