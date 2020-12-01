import numpy as np
from scipy.fft import fft,ifft
from astropy import units as u
from time import perf_counter as timer
tic= timer()
# ============== 01. make CCD coordinate =======================
x = np.linspace(1, 1000, 1000, endpoint=True)  # coordinate of CCD
x = x * 1 / np.max(x) * 10 ** -2  # 1cm by 1cm CCD
center_CCD = (x[0] + x[-1]) / 2  # define center of CCD
x = x - center_CCD

xv, yv = np.meshgrid(x, x)
print('len.xv',len(xv))
xv = np.ravel(xv)
yv = np.ravel(yv)

# ============== 02. make lens coordinate =======================
d = 1 * 10 ** -3  # diameter of lens

x_prime = np.linspace(1, 100, 100, endpoint=True)  # coordinate of lens
x_prime = x_prime * 1 / 100 * d  # physical length
center_pupil = (x_prime[0] + x_prime[-1]) / 2  # define center of lens

x_prime = x_prime - center_pupil
xv_prime, yv_prime = np.meshgrid(x_prime, x_prime)


# ============== 03. extract index and dA calculation =======================
r = d/2  # radius of lens
ind = np.where(xv_prime**2 + yv_prime**2 < r**2)
xv_prime = xv_prime[ind]
yv_prime = yv_prime[ind]
print('len(ind[0])',len(ind[0]))
s = np.pi * r**2 / len(ind[0])


# ============== 04. define params =======================
wavelength = 400. * 10 ** -9
k_0 = 2 * np.pi / wavelength  # wavenumber in vacuum

f = 50. * 10 ** -3  # focal length of lens = R/(n-1)
D = 50. * 10 ** -3  # distance to CCD

i = complex(0, 1)  # imaginary number i

# ============== 05. define funcs =======================

def fun1(x1, y1,a,b):
    r = (x1 - a) ** 2 + (y1 - b) ** 2
    val2 = - np.pi * r / (wavelength * D)
    return val2
def fun2(x,y):
    return  k_0 * (x ** 2 + y ** 2) / (2 * f)


def U(x1, y1):

    pre1 = fun1(xv_prime, yv_prime,x1,y1)
    pre2 = fun2(xv_prime,yv_prime)
    pre = pre1 + pre2
    int = np.exp(i*pre)
    dA = s
    U = np.sum(int)*s
    I = abs(U)**2

    return I
# ============== 06. make U(x,y) array =======================
results = np.zeros(0)
for i in range(len(xv)):
    x1 = xv[i]
    y1 = yv[i]
    result = U(x1,y1)
    results = np.append(results,result)
    print(i/len(xv)*100)

print(results.shape)

results = results.reshape(1000,1000)

# ============== 07. plotting =======================
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
hdu = fits.PrimaryHDU(results)
hdul = fits.HDUList([hdu])
hdul.writeto('numerical_int_D_50mm_wl_400mm_1000x1000.fits')



im = plt.imshow(results,cmap='gray',origin='lower',norm=LogNorm())
plt.colorbar(im)
plt.title('I')
plt.show()