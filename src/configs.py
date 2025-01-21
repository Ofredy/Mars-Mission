import numpy as np


MARS_G_CONST = 42828.3

# RN DCM
r3 = np.array([0, 1, 0])
r1 = np.array([-1, 0, 0])
r2 = np.cross(r3, r1)
rn_dcm = np.column_stack((r1, r2, r3))
