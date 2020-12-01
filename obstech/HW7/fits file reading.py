from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
hdulist = fits.open('0.5s_PSF_2000x2000_100m.fits')
image_05 = hdulist[0].data

hdulist = fits.open('15s_PSF_2000x2000_100m.fits')
image_15 = hdulist[0].data

hdulist = fits.open('30s_PSF_2000x2000_100m.fits')
image_30 = hdulist[0].data

hdulist = fits.open('60s_PSF_2000x2000_100m.fits')
image_60 = hdulist[0].data

hdulist = fits.open('120s_PSF_2000x2000_100m.fits')
image_120 = hdulist[0].data

x1 = 950
x2 = 1050
color = 'gist_ncar'

plt.figure(1)
plt.imshow(image_05,cmap=color,norm=LogNorm())
# plt.imshow(image_05,cmap=color)
plt.xlim(x1,x2)
plt.ylim(x1,x2)
plt.tight_layout()

plt.figure(2)
plt.imshow(image_15,cmap=color,norm=LogNorm())
# plt.imshow(image_15,cmap=color)
plt.xlim(x1,x2)
plt.ylim(x1,x2)
plt.tight_layout()

plt.figure(3)
plt.imshow(image_30,cmap=color,norm=LogNorm())
# plt.imshow(image_30,cmap=color)
plt.xlim(x1,x2)
plt.ylim(x1,x2)
plt.tight_layout()

plt.figure(4)
plt.imshow(image_60,cmap=color,norm=LogNorm())
# plt.imshow(image_60,cmap=color)
plt.xlim(x1,x2)
plt.ylim(x1,x2)
plt.tight_layout()

plt.figure(5)
plt.imshow(image_120,cmap=color,norm=LogNorm())
# plt.imshow(image_120,cmap=color)
plt.xlim(x1,x2)
plt.ylim(x1,x2)
plt.tight_layout()

plt.show()