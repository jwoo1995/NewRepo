import numpy as np
from astropy import units as u
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
print(coord_CCD[0])
print(len(coord_CCD))

from scipy.fft import fft,ifft

wavelength = 600 * 10 ** -9 * u.m
wavelength = wavelength.value
k = 2. * np.pi / wavelength
d1 = 0.1 * 10 ** -3 * u.m  # diameter of aperture
d1 = d1.value
f = 20 * 10 ** -3 * u.m
f = f.value  # focal length
j = complex(0,1)
def fun1(x1, y1,a,b):
    r = (x1 - a) ** 2 + (y1 - b) ** 2
    val1 = 1 / r * np.exp(j * k * r)
    val2 = np.exp(-j * np.pi * r / (wavelength * f))
    return val1, val2
indd= coord_CCD[0]
a=indd[0]
b=indd[1]
x1 = np.linspace(1,100,100,endpoint=True)
pre=fun1(x1,x1,a,b)
post1= fft(pre[0])
post2= fft(pre[1])
post=fft(pre)

print('post',post)
print('post1',post1)
print('post2',post2)
print(post[0]-post1)