from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from astropy import units as u
import pandas as pd
# ===================================== 01. load file ========================================
hdulist = fits.open('hst_pupil.fits')
image = hdulist[0].data

# ===================================== 02. PSF ========================================
U = np.fft.ifftshift(np.fft.ifft2(image))
I = abs(U)**2
print(np.where(np.max(I)==I))

# ===================================== 03. slicing ========================================
J_y = I[500:501,:]
J_y = np.ravel(J_y)
J_x = np.linspace(0, 999, 1000,endpoint=True)

# ===================================== 04. interpolation ========================================
Max = np.max(J_y)
# J_y[J_y == Max] = 0
# Max2 = np.max(J_y)

# distance = (Max - (Max/ 2)) / (Max - Max2)
from scipy.interpolate import CubicSpline
cs = CubicSpline(J_x,J_y)
xs = np.linspace(0, 999, 90000,endpoint=True)

from scipy.interpolate import interp1d
cubic = cs(xs)
ind = np.where(abs(Max/2 - cubic) < 10 ** -4)
print(ind)
dist = 2 * np.mean(abs(len(xs)/2 - ind[0])) / 90
print(dist)

cs2 = interp1d(J_x,J_y)


plt.scatter(xs/2,cs(xs),label = 'Cubic spline interpolation', c='g', s=0.5)
plt.xlim(200,300)
# plt.scatter(xs,cs2(xs),label = 'linear interpolation', c='g', s=0.05)

# plt.scatter(J_x,J_y, label = 'PSF')
# plt.legend()
plt.show()