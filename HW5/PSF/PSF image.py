from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from astropy import units as u
import pandas as pd
hdulist = fits.open('hst_pupil.fits')
image = hdulist[0].data
U = np.fft.ifftshift(np.fft.ifft2(image))
I = abs(U)**2

# Max = np.max(I)
# I[I == Max] = 0
# Max2 = np.max(I)
# print(Max, Max2)

plt.figure(1)
im = plt.imshow(I,cmap='gray',origin='lower',norm=LogNorm())
plt.figure(2)
im2 = plt.imshow(I,cmap='gray',origin='lower')

# im2= plt.imshow(HM,cmap='gray',origin='lower',norm=LogNorm())
plt.title('I')
plt.show()
