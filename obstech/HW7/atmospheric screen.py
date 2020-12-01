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
U = np.fft.fftshift(np.fft.fft2(image_array1.real))
I1 = abs(U) ** 2
plt.figure(1)
plt.imshow(I1,cmap='jet',norm=LogNorm())
plt.figure(2)
plt.imshow(image_array1.real,cmap='jet')
plt.tight_layout()
plt.show()

# image_array2 = np.fft.ifft2(powerspec_array1*Phase_array*np.sqrt(0.023) * (2000)**(5/6))
# image_array3 = 0.023 * (1/r_0)**(5/3) * (np.fft.ifft2(powerspec_array2*Phase_array))
# image1 = image_array1.real
# print(np.max(image1),np.min(image1))
# image2 = image_array2.imag
# image3 = image_array3.imag
# # print(image)
# plt.figure(1)
# plt.imshow(image1,cmap='jet')
# plt.figure(2)
# plt.imshow(image2,cmap='jet')
# plt.figure(3)
# plt.imshow(image3,cmap='jet',interpolation='spline16')
# plt.show()