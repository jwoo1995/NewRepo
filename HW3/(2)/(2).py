import numpy as np
from scipy.fft import fft,ifft
from astropy import units as u
from time import perf_counter as timer
tic= timer()
# ============== 01. make coordinate =======================
d1 = 0.1 * 10 ** -3 * u.m  # diameter of aperture
d1 = d1.value
f = 20 * 10 ** -3 * u.m
f = f.value  # focal length
x_prime = np.linspace(1, 100, 100, endpoint=True)  # coordinate of aperture
x_prime = x_prime * 1 / 100 * d1  # physical length

center_pupil = (x_prime[0] + x_prime[-1]) / 2
x_prime = x_prime - center_pupil
xv_prime, yv_prime = np.meshgrid(x_prime,x_prime)



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
#coord_CCD=(list(zip(coord_CCD)))
j = complex(0., 1.)


def fun1(x1, y1,a,b):
    r = (x1 - a) ** 2 + (y1 - b) ** 2
    val1 = 1 / r * np.exp(j * k * r)
    val2 = np.exp(-j * np.pi * r / (wavelength * f))
    return val1, val2


def U(coord):  # a,b를 여기에 포함시킬 수 있을 듯
    a = coord[0]
    b = coord[1]
    pre = fun1(xv_prime, yv_prime,a,b)

    U1 = fft(pre)
    U3 = ifft(U1[0]*U1[1])
    I = abs(U3)**2

    return I
#print('s', s)
"""
from multiprocessing import Pool

if __name__=='__main__':
    with Pool(4) as p:
        results = p.starmap(U,[(row) for row in coord_CCD])
        np.savetxt('(2)-i',results)

tac1 = timer()
"""
results = np.zeros(0)
for i in range(len(coord_CCD)):
    I1 = U(coord_CCD[i])
    results = np.append(results, I1)
    print(i/10**6*10**2)

"""
for row in coord_CCD:

    I1 = U(row)
    results = np.append(results,I1)
"""
np.savetxt('(2)-i',results)
