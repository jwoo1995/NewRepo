import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
I = np.loadtxt('(2)-i')
print(len(I))

I = I.reshape((1000,1000))
from astropy.io import fits
hdu = fits.PrimaryHDU(I)
hdul = fits.HDUList([hdu])
hdul.writeto('(2).fits')


im = plt.imshow(I,cmap='gray',origin='lower',norm=LogNorm())
plt.colorbar(im)
plt.show()
