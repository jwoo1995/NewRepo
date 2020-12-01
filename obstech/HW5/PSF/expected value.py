import numpy as np
wavelength = 600 * 10**-9
d = 2.4
theta_rad = 1.03 * wavelength / d
theta_arcsec = theta_rad * 180 / np.pi * 3600
print(theta_arcsec)

