import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# =================== 01. random phase =================
ran_phase = np.random.uniform(low=0.0, high=2*np.pi, size=(2000,2000))
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
# powerspec_array = (f_array + (2/L_0)**2)**(-11/12)
powerspec_array = (f_array + (6)**2)**(-11/6)

# =========== 04. IFT and real image =================

# image_array = np.sqrt(2) * np.sqrt(0.023) * (L/r_0)**(5/6) * np.fft.fftshift(np.fft.ifft2(powerspec_array*Phase_array))
image_array = 0.023 * (1/r_0)**(5/3) * np.fft.ifft2(powerspec_array*Phase_array)
image = image_array.real
# print(image)
plt.imshow(image,cmap='plasma',vmin=np.min(image),vmax=np.max(image))
plt.show()
