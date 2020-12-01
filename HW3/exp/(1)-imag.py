import numpy as np
from scipy import integrate
from astropy import units as u

# ============== 01. make coordinate =======================
d1 = 0.1 * 10 ** -3 * u.m  # diameter of aperture
d1 = d1.value
f = 20 * 10 ** -3 * u.m
f = f.value  # focal length
x_prime = np.linspace(1, 100, 100, endpoint=True)  # coordinate of aperture
x_prime = x_prime * 1 / 100 * d1  # physical length
# print(x_prime)

center_pupil = (x_prime[0] + x_prime[-1]) / 2
x_prime = x_prime - center_pupil
r_pupil = d1 / 2
wavelength = 600 * 10 ** -9 * u.m
wavelength = wavelength.value
k = 2. * np.pi / wavelength

x = np.linspace(1, 1000, 1000, endpoint=True)  # coordinate of CCD
y = np.linspace(1, 1000, 1000, endpoint=True)
center_CCD = (x[0] + x[-1]) / 2

x = x * 1 / np.max(x) * 10 ** -2
y = y * 1 / np.max(y) * 10 ** -2
x = x - center_CCD
y = y - center_CCD

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
        r = np.sqrt(f ** 2. + (x1 - a) ** 2. + (y1 - b) ** 2.)
        val = 1 / r * np.exp(j * k * r)
        return val.imag
    U2 = integrate.dblquad(fun1, np.min(x_prime), np.max(x_prime), lambda x: -np.sqrt(r_pupil ** 2 - x ** 2),
                              lambda x: np.sqrt(r_pupil ** 2 - x ** 2))
    return U2[0]

from multiprocessing import Pool

if __name__=='__main__':
    with Pool(6) as p:
        results = p.starmap(U,[(row) for row in coord_CCD])
        np.savetxt('(1)-multi_imag',results)



