import numpy as np
with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0
import matplotlib.pyplot as plt
import scipy
from scipy import integrate
import pandas
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
wavelength = 600 * 10 ** -3 * u.m
wavelength = wavelength.value
k = 2. * np.pi / wavelength

x = np.linspace(1, 1000, 1000, endpoint=True)  # coordinate of CCD
y = np.linspace(1, 1000, 1000, endpoint=True)
center_CCD = (x[0] + x[-1]) / 2

x = (x - center_CCD) * 1 / np.max(x) * 10 ** -2
y = (y - center_CCD) * 1 / np.max(y) * 10 ** -2

xv, yv = np.meshgrid(x, y)
xv = np.ravel(xv)
yv = np.ravel(yv)

def U(x_prime, x_CCD, y_CCD):  # a,b를 여기에 포함시킬 수 있을 듯
    U_zero = np.zeros(0)
    j = complex(0., 1.)
    for i in range(len(x_CCD)):
        a = x_CCD[i]
        b = y_CCD[i]
        def fun1(x1, y1):
            r = np.sqrt(f ** 2. + (x1 - a) ** 2. + (y1 - b) ** 2.)
            val = 1 / r * np.exp(j * k * r)
            return val.real
        def fun2(x1, y1):
            r = np.sqrt(f ** 2. + (x1 - a) ** 2. + (y1 - b) ** 2.)
            val = 1 / r * np.exp(j * k * r)
            return val.imag

        U1 = integrate.dblquad(fun1, np.min(x_prime), np.max(x_prime), lambda x: -np.sqrt(r_pupil ** 2 - x ** 2),
                              lambda x: np.sqrt(r_pupil ** 2 - x ** 2))
        U2 = integrate.dblquad(fun2, np.min(x_prime), np.max(x_prime), lambda x: -np.sqrt(r_pupil ** 2 - x ** 2),
                               lambda x: np.sqrt(r_pupil ** 2 - x ** 2))
        U_zero = np.append(U_zero, U1[0]+1j*U2[0])
        print(i/len(x_CCD)*100)
    I = abs(U_zero) ** 2
    return I


I = U(x_prime, xv, yv)
print(I)