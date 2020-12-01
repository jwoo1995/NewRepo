import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
screen_pixel = 2000
pupil_pixel= 16

def circle_filter(screen_pixel,pupil_pixel):
    x = np.linspace(1,screen_pixel,screen_pixel,endpoint=True)
    center = (np.min(x)+np.max(x))/2
    x = x - center
    xv, yv = np.meshgrid(x,x)
    radius = pupil_pixel/2
    fil_ind = np.where(xv**2+yv**2 <= radius**2)
    return fil_ind

I = np.zeros((screen_pixel,screen_pixel))
I[circle_filter(screen_pixel,pupil_pixel)]=1
plt.imshow(I,cmap='gray')
plt.tight_layout()
plt.show()