from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import pandas as pd
hdulist = fits.open('hst_pupil.fits')
image = hdulist[0].data

plt.imshow(image,cmap='gray')
plt.show()