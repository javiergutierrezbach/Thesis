import numpy as np
import torch
import math

epsilon = 1e-2
L1 = 2.4
L2 = 2.1
M = 1.2705e-10

spacing = 2 * (epsilon - M)/ (L1 + L2)

print(spacing)


pos_limit = 2.2
vel_limit = 0.55
force_limit = 1.5

num_points = (math.floor(2.2 * 2 / spacing) ** 2) * (math.floor(0.55 * 2 / spacing) ** 2) 
print(num_points)
print(math.log10(num_points))

num_points = 0


for x in np.arange(-pos_limit, pos_limit, spacing):
    for y in np.arange(-pos_limit, pos_limit, spacing):
        for vx in np.arange(-vel_limit, vel_limit, spacing):
            for vy in np.arange(-vel_limit, vel_limit, spacing):
                for fx in np.arange(-force_limit, force_limit, spacing):
                    for fy in np.arange(-force_limit, force_limit, spacing):
                        num_points += 1
                        
print(num_points)