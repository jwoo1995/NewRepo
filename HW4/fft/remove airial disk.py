import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from time import perf_counter as timer
import numpy.fft as fft
from matplotlib.colors import LogNorm
tic = timer()
# ============== 01. make CCD coordinate =======================
x = np.linspace(1, 2000, 2000, endpoint=True)  # coordinate of CCD
x = x * 2 / np.max(x) * 10 ** -2  # 2cm by 2cm CCD
center_CCD = (x[0] + x[-1]) / 2  # define center of CCD
x = x - center_CCD

xv, yv = np.meshgrid(x, x)
CCD_indx = np.where( (-0.5 * 10**-2 < xv) & (xv < 0.5 * 10**-2) & (-0.5 * 10**-2 < yv) & (yv < 0.5 * 10**-2))
# CCD_indy = np.where( (-0.5 * 10**-2 < yv) & (yv < 0.5 * 10**-2))

new_xv = xv[CCD_indx]
new_yv = yv[CCD_indx]

new_xv = new_xv.reshape(1000,1000)
new_yv = new_yv.reshape(1000,1000)

# print(new_xv.shape)
# print(new_xv)
# print(new_yv.shape)


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
from astropy.io import fits
H = np.zeros((2000,2000))
h = h(new_xv,new_yv)
h = np.ravel(h)


print(CCD_indx[0].shape)
# H[CCD_indx] = h.real + complex(0,1)*h.imag
H[CCD_indx] = h.real + complex(0,1)*h.imag
print('H.shape',H.shape)



U = U(xv, yv)
U[ind] = 0

print(abs(H))



from scipy import signal


I = signal.fftconvolve(U,H,mode='same')


# K = I[CCD_indx]
# K = K.reshape(1000,1000)

K = I
print('K.shape',K.shape)
K = abs(K) **2


# ============== 07. plotting =======================
from astropy.io import fits
hdu = fits.PrimaryHDU(abs(H))
hdul = fits.HDUList([hdu])
hdul.writeto('H.fits')


plt.figure(1)
im = plt.imshow(K,cmap='gray',origin='lower',norm=LogNorm())
plt.colorbar(im)

plt.figure(2)
im2 = plt.imshow(abs(H),cmap='gray',origin='lower',norm=LogNorm())
plt.colorbar(im2)

plt.figure(3)
im3 = plt.imshow(abs(U),cmap='gray',origin='lower',norm=LogNorm())
plt.colorbar(im3)


plt.show()



