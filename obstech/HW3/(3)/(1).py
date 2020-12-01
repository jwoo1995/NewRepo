import numpy as np
from scipy import integrate
from astropy import units as u
from time import perf_counter as timer
tic= timer()
# ============== 01. make coordinate =======================
d1 = 0.1 * 10 ** -3 * u.m  # diameter of aperture
d1 = d1.value
f = 10 * 10 ** -3 * u.m
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
print('s',s)


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

from time import perf_counter as timer
def U(coord):  # a,b를 여기에 포함시킬 수 있을 듯
    tic = timer()
    j = complex(0., 1.)
    a = coord[0]
    b = coord[1]
    def fun1(x1, y1):
        r = np.sqrt(f ** 2. + (x1 - a) ** 2. + (y1 - b) ** 2.)
        val1 = 1 / r * np.cos(k * r)
        val2= 1 / r * np.sin(k*r)
        return val1, val2

    pre = fun1(xv_prime, yv_prime)
    U1 = np.sum(pre[0]*s)
    U2 = np.sum(pre[1]*s)
    I = abs(U1 + j*U2)**2

    return I
tac = timer()
print(tac-tic)

from multiprocessing import Pool

if __name__=='__main__':
    with Pool(6) as p:
        results = p.starmap(U,[(row) for row in coord_CCD])
        np.savetxt('(1)-i',results)

