import numpy as np
from matplotlib.colors import LogNorm
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
import pandas
from astropy import units as u

imag = np.loadtxt('(1)-multi_imag')
real = np.loadtxt('(1)-multi_real')
print(len(real))
print(len(imag))
real = real[:,0]

j = complex(0., 1.)
U = real + j*imag
I = abs(U)**2

I = I.reshape((1000,1000))

from astropy.io import fits
hdu = fits.PrimaryHDU(I)
hdul = fits.HDUList([hdu])
hdul.writeto('new1.fits')



im = plt.imshow(I,cmap='gist_rainbow',origin='lower',norm=LogNorm())
plt.colorbar(im)
plt.show()
