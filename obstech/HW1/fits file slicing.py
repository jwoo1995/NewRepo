from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
"============================== 01. DATA 가져 오기 ============================="
hdulist = fits.open('Moon.fits')

image = hdulist[0].data
# print(image.shape)

x = np.linspace(1,image.shape[1],image.shape[1])
y = np.linspace(1,image.shape[0],image.shape[0])

"==================== 02. 달에 반지름에 해당하는 pixel 수 구하기 ==================="
x_median = int(np.round(np.median(x)))
y_median = int(np.round(np.median(y)))
slice = image[:,x_median-1:x_median]
ind = slice > 0
new_slice = slice[ind]
diameter_pixel = new_slice.shape[0] # diameter pixel = diameter of moon