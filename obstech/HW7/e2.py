import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# =================== 01. random phase =================
ran_phase = np.random.uniform(low=-np.pi, high=np.pi, size=(2000,2000))
j= complex(0,1)
Phase_array = np.exp(j*ran_phase)

# =================== 02. frequency =================

f1 = np.linspace(0,1000,1001,endpoint=True)
f2 = np.linspace(1,999,999,endpoint=True)
f = np.append(f1,f2[::-1])

f_x, f_y = np.meshgrid(f,f)
f_array = f_x**2 + f_y**2
# print(f2[::-1])
# print(f)

# =========== 03. power spectrum array =================
L_0 = 25
L = 1000
r_0 = 0.2
powerspec_array1 = (f_array + (L/L_0)**2)**(-11/12)
# powerspec_array2 = (f_array + (1/L_0)**2)**(-11/6)

# =========== 04. IFT and real image =================
image_array1 = np.sqrt(2) * np.sqrt(0.023) * (L/r_0)**(5/6) * np.fft.ifft2(powerspec_array1*Phase_array)
image1 = image_array1.real

# image_1st = image1[:, 1:2000]
# print(image_1st.shape)
# image_2nd = image1[:, 0:1]
# print(image_2nd.shape)
# image_exp = np.append(image_1st, image_2nd, axis=1)
# print(image_exp.shape)
# print(len(image1[0,:]))
# print(image1[999-8:1000+8,999-8:1000+8])
# print(image1[:, 0:1])
# print(image_exp[:, 1999:2000])
# print(image1[:, 0:1] - image_exp[:, 1999:2000])
# ========= 05. Periodic Boundary condition ============
def circle_filter(screen_pixel,pupil_pixel):
    x = np.linspace(1,screen_pixel,screen_pixel,endpoint=True)
    center = (np.min(x)+np.max(x))/2
    x = x - center
    xv, yv = np.meshgrid(x,x)
    radius = pupil_pixel/2
    fil_ind = np.where(xv**2+yv**2 > radius**2)
    return fil_ind

I = np.zeros((2000, 2000))
def PSF(exptime,image,I):
    timestep = 0.005
    k = exptime/timestep
    k = int(k)
    print(k)
    pupil_size = 8
    screen_pixel = len(image[0,:])
    pupil_pixel = screen_pixel * pupil_size / L
    for i in range(0,k):
        timestep_num = i
        if timestep_num%20 == 7 or timestep_num%20 == 14 or timestep_num%20 == 0:
            print(timestep_num)
            image_1st = image[:, 0:screen_pixel-1]
            image_2nd = image[:, screen_pixel-1:screen_pixel]
            image = np.append(image_2nd,image_1st,axis=1)
            # ind = screen_pixel / 2 - pupil_pixel / 2
            # ind = int(ind)
            # print(ind,ind+16)
            # pupil = image[ind:ind + 16, ind:ind + 16]
            U = np.fft.fftshift(np.fft.fft2(image))
            I1 = abs(U) ** 2
            I = I + I1
    print(np.min(I))
    return I
# plt.figure(1)
# plt.imshow(PSF(exptime=0.5, image= image1, I=I),cmap='jet',norm=LogNorm())
# plt.show()
from astropy.io import fits
# hdu = fits.PrimaryHDU(PSF(exptime=15, image= image1, I=I))
# hdul = fits.HDUList([hdu])
# hdul.writeto('15s_PSF_2000x2000.fits',overwrite=True)
#
hdu = fits.PrimaryHDU(PSF(exptime=30, image= image1, I=I))
hdul = fits.HDUList([hdu])
hdul.writeto('30s_PSF_2000x2000.fits',overwrite=True)
#
# hdu = fits.PrimaryHDU(PSF(exptime=60, image= image1, I=I))
# hdul = fits.HDUList([hdu])
# hdul.writeto('60s_PSF_2000x2000.fits',overwrite=True)
#
# hdu = fits.PrimaryHDU(PSF(exptime=120, image= image1, I=I))
# hdul = fits.HDUList([hdu])
# hdul.writeto('120s_PSF_2000x2000.fits',overwrite=True)

# hdu = fits.PrimaryHDU(PSF(exptime=120, image= image1))
# hdul = fits.HDUList([hdu])
# hdul.writeto('120s_PSF.fits',overwrite=True)
