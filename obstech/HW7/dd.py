import numpy as np
I = np.zeros((4,4))
I1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8],[9, 10, 11, 12],[13, 14, 15, 16]])
for i in range(0,5):
    if i%3 == 1:
        img1 = I1[:,3:4]
        img2 = I1[:,0:3]
        I1 = np.append(img1,img2, axis=1)
        I = I + I1
        print(np.min(I))
print(I)