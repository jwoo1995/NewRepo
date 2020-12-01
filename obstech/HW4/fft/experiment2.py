import numpy as np

x = np.linspace(1, 1000, 1000, endpoint=True)  # coordinate of CCD
x = x * 1 / np.max(x) * 10 ** -2  # 1cm by 1cm CCD
center_CCD = (x[0] + x[-1]) / 2  # define center of CCD
x = x - center_CCD
xv, yv = np.meshgrid(x, x)

d = 1 * 10 ** -3  # diameter of lens

x_prime = np.linspace(1, 100, 100, endpoint=True)  # coordinate of lens
x_prime = x_prime * 1 / 100 * d  # physical length
center_pupil = (x_prime[0] + x_prime[-1]) / 2  # define center of lens

x_prime = x_prime - center_pupil
xv_prime, yv_prime = np.meshgrid(x_prime, x_prime)

# ============== 03. extract index =======================
r_pupil = d / 2  # radius of lens
ind_prime = np.where(xv_prime ** 2. + yv_prime ** 2. > r_pupil ** 2)

# ============== 04. define params =======================
wavelength = 400. * 10 ** -9
k_0 = 2 * np.pi / wavelength  # wavenumber in vacuum

f = 50. * 10 ** -3  # focal length of lens = R/(n-1)
D = 50. * 10 ** -3  # distance to CCD

i = complex(0, 1)  # imaginary number i


# ============== 05. define funcs =======================
def h(x, y):
    return np.exp(-i * np.pi * (x ** 2 + y ** 2) / (wavelength * D))

def U(x,y):
    r = np.sqrt(x**2 + y**2)
    return 1/r * np.exp(i* k_0 * r)

print (h(xv, yv).shape)
H = np.fft.fft2(h(xv, yv))
U = np.fft.fft2(U(xv, yv))
I = np.fft.ifft2(H * U)

I = abs(I) ** 2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
im = plt.imshow(I,cmap='gray',origin='lower',norm=LogNorm())
plt.colorbar(im)
plt.show()
from astropy.io import fits
hdu = fits.PrimaryHDU(I)
hdul = fits.HDUList([hdu])
hdul.writeto('s.fits')
