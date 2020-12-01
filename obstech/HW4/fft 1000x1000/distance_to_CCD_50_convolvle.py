import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from time import perf_counter as timer
import numpy.fft as fft
from matplotlib.colors import LogNorm
tic = timer()
# ============== 01. make CCD coordinate =======================
x = np.linspace(1, 1000, 1000, endpoint=True)  # coordinate of CCD
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
wavelength = 800. * 10 ** -9
k_0 = 2 * np.pi / wavelength  # wavenumber in vacuum

f = 50. * 10 ** -3  # focal length of lens = R/(n-1)
D = 80. * 10 ** -3  # distance to CCD

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
I = signal.fftconvolve(H,U,mode='same')
I = abs(I) **2

# U = fft.fft2(U)
# I = fft.ifft2(H * U)
# I = abs(I) **2


# ============== 07. plotting =======================
from astropy.io import fits
hdu = fits.PrimaryHDU(I)
hdul = fits.HDUList([hdu])
hdul.writeto('fft_convolve_fullmode_D_80mm_wl_800mm_1000x1000.fits')





"""
I = I.reshape((1000,1000))

from astropy.io import fits
hdu = fits.PrimaryHDU(I)
hdul = fits.HDUList([hdu])
hdul.writeto('s.fits')



im = plt.imshow(I,cmap='gray',origin='lower',norm=LogNorm())
plt.colorbar(im)
plt.show()



x_prime = x_prime - center_pupil
xv_prime, yv_prime = np.meshgrid(x_prime,x_prime)


d = 1 * 10 ** -3  # diameter of aperture

f = 10 * 10 ** -3 * u.m
f = f.value  # focal length
x_prime = np.linspace(1, 100, 100, endpoint=True)  # coordinate of aperture
x_prime = x_prime * 1 / 100 * d1  # physical length

r_pupil = d1 / 2
ind_prime = np.where(xv_prime**2. + yv_prime**2.<=r_pupil**2)
xv_prime = xv_prime[ind_prime]
yv_prime = yv_prime[ind_prime]
xv_prime = np.ravel(xv_prime)
yv_prime = np.ravel(yv_prime)
s= np.pi * r_pupil**2/len(xv_prime)

wavelength = 600 * 10 ** -9 * u.m
wavelength = wavelength.value
k = 2. * np.pi / wavelength

x = np.linspace(1, 1000, 1000, endpoint=True)  # coordinate of CCD
y = np.linspace(1, 1000, 1000, endpoint=True)


x = x * 1 / np.max(x) * 10 ** -2
y = y * 1 / np.max(y) * 10 ** -2
center_CCD = (x[0] + x[-1]) / 2
x= x - center_CCD
y= y - center_CCD
xv, yv = np.meshgrid(x, y)
xv = np.ravel(xv)
yv = np.ravel(yv)

xv = xv.reshape((len(xv),1))
yv = yv.reshape((len(yv),1))
coord_CCD = np.concatenate((xv,yv),axis=1)
coord_CCD=(list(zip(coord_CCD)))

def U(coord):  # a,b를 여기에 포함시킬 수 있을 듯
    j = complex(0., 1.)
    a = coord[0]
    b = coord[1]
    def fun1(x1, y1):
        r = (x1-a)**2 + (y1-b)**2
        val1 = 1 / f * np.cos(- np.pi* r/(wavelength*f))
        val2 = 1 / f * np.sin(- np.pi* r/(wavelength*f))
        return val1, val2

    pre = fun1(xv_prime, yv_prime)
    U1 = np.sum(pre[0]*s)
    U2 = np.sum(pre[1]*s)
    I = abs(U1 + j*U2)**2

    return I
print('s', s)

from multiprocessing import Pool

if __name__=='__main__':
    with Pool(6) as p:
        results = p.starmap(U,[(row) for row in coord_CCD])
        np.savetxt('(2)-i',results)

tac1 = timer()
"""
