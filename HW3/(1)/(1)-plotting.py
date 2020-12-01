import numpy as np
from matplotlib.colors import LogNorm
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
import pandas
from astropy import units as u



I = np.loadtxt('(1)-I')
print(len(I))

I = I.reshape((1000,1000))
"""
from astropy.io import fits
hdu = fits.PrimaryHDU(I)
hdul = fits.HDUList([hdu])
hdul.writeto('s.fits')

"""

im = plt.imshow(I,cmap='gray',origin='lower',norm=LogNorm())
plt.colorbar(im)
plt.show()
