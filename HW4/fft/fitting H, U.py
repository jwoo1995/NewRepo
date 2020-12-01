import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from time import perf_counter as timer
import numpy.fft as fft
from matplotlib.colors import LogNorm
tic = timer()
# ============== 01. make CCD coordinate =======================
x = np.linspace(1, 2000, 2000, endpoint=True)  # coordinate of CCD
x = x * 1 / np.max(x) * 10 ** -2  # 1cm by 1cm CCD
center_CCD = (x[0] + x[-1]) / 2  # define center of CCD
x = x - center_CCD

xv, yv = np.meshgrid(x, x)


# ============== 02. make lens coordinate =======================
d = 1 * 10 ** -3  # diameter of lens

x_prime = np.linspace(1, 100, 100, endpoint=True)  # coordinate of lens
x_prime = x_prime * 1 / 100 * d  # physical length
center_pupil = (x_prime[0] + x_prime[-1]) / 2  # define center of lens

x_prime = x_prime - center_pupil
xv_prime, yv_prime = np.meshgrid(x_prime, x_prime)

# ============== 03. extract index =======================
r = d/2  # radius of lens
ind = np.where(xv**2 + yv**2 > r**2)

# ============== 04. define params =======================
wavelength = 400. * 10 ** -9
k_0 = 2 * np.pi / wavelength  # wavenumber in vacuum

f = 50. * 10 ** -3  # focal length of lens = R/(n-1)
D = 50. * 10 ** -3  # distance to CCD

i = complex(0, 1)  # imaginary number i


# ============== 05. define funcs =======================
def h(x, y):
    return np.exp(-i * np.pi * ((x) ** 2 + (y) ** 2) / (wavelength * D))


def U(x, y):  # h_0 = 1
    return np.exp(i * k_0 * (x ** 2 + y ** 2) / (2 * f))


# ============== 06. make U(x,y) array =======================
# H = fft.fft2(h(xv, yv))
H = h(xv, yv)
U = U(xv, yv)
U[ind] = 0
from astropy.io import fits
hdu = fits.PrimaryHDU(U.real)
hdul = fits.HDUList([hdu])
hdul.writeto('U.real_5000x5000.fits')

"""
from astropy.io import fits
hdu = fits.PrimaryHDU(H.real)
hdul = fits.HDUList([hdu])
hdul.writeto('H.real_5000x5000.fits')

# ============== 07. fitting =======================
plt.figure(1)
im = plt.imshow(abs(U),cmap='gray',origin='lower',norm=LogNorm())
plt.title(r'|U(x,y,0)|')

plt.figure(2)
im2 = plt.imshow(H.real,cmap='gray',origin='lower',norm=LogNorm())

plt.title(r'|h(x,y,d)|')
plt.show()
"""
