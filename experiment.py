import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
mean = [0, 0]
cov = [[1, 1], [1, 2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T
print('x',x)
print('y',y)

data = np.vstack([x, y])
print('data',data)
print('data.shape',data.shape)
kde = gaussian_kde(data)
print('kde',kde)


# evaluate on a regular grid
xgrid = np.linspace(-3.5, 3.5, 40)
ygrid = np.linspace(-6, 6, 40)
Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(), Ygrid.ravel()]))
print('Z',Z)
print('Z.shape',Z.shape)
re = Z.reshape(Xgrid.shape)
print('Z.reshape',re.shape)
# Plot the result as an image
plt.imshow(Z.reshape(Xgrid.shape),
           origin='lower', aspect='auto',
           extent=[-3.5, 3.5, -6, 6],
           cmap='Blues')
cb = plt.colorbar()
cb.set_label("density")
plt.show()
noise = np.random.normal(0,1,100)

# print('noise',noise)