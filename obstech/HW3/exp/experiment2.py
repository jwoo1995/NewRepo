import numpy as np
xv= np.array([1,2,3])
yv = np.array([4,5,6])

xv = xv.reshape((3,1))

yv = yv.reshape((3,1))
coord_CCD = np.concatenate((xv,yv),axis=1)


coord_CCD=(list(zip(coord_CCD)))


from multiprocessing import Pool

def U(coord):
    print(np.where(coord==coord_CCD)[0])
    return coord[0]+coord[1]


if __name__=='__main__':
    with Pool(6) as p:
        results = p.starmap(U,[(row) for row in coord_CCD])
        print(results)
