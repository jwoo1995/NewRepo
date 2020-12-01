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
print(I[500, 500])
print(I[500, 501])
print(I[501, 501] * np.sqrt(2))
# ===================================== 03. graph ========================================
print(I.shape)
J_y = I[500:501,:]
J_y = np.ravel(J_y)
J_x = np.linspace(0,999,1000,endpoint=True)
plt.plot(J_x,J_y)
# plt.xlim(498, 502)
# ticks = [round(J_y[499],4),round(J_y[500],4)]
# print(np.max(J_y))
# plt.yticks(ticks, labels=ticks, rotation = 'horizontal')
# plt.tight_layout()
plt.ylabel('I (intensity)')
plt.show()