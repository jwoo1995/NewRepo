from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

hdulist = fits.open('Moon.fits')
image = hdulist[0].data

print(image[673,1807],image[1416,739])  # y(row),x(column)의 형태로 들어가는 것
plt.imshow(image, cmap='gray', origin='lower')
plt.colorbar()
plt.show()