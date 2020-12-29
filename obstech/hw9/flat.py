import numpy as np
import io

import astropy.io.fits as fits
import matplotlib.pyplot as plt
# =============== 01. file =======================
sex_list = open('sex.lis','r')
# sex_list = sex_list.read()

seg_list = open('seg.lis','r')
# seg_list = seg_list.read()

sex_arr = np.zeros(0)
for line in sex_list.readlines():
    line = line.strip('\n')
    sex_arr = np.append(sex_arr,line)
sex_list.close()

seg_arr = np.zeros(0)
for line in seg_list.readlines():
    line = line.strip('\n')
    seg_arr = np.append(seg_arr, line)
seg_list.close()

print('seg_arr',seg_arr)
print('sex_arr',sex_arr)
# print('len of seg_arr', len(seg_arr))
# print('len of sex_arr', len(sex_arr))
# print('seg_arr[1]', seg_arr[1])
fits.open(sex_arr[1])

for i in range(0,len(seg_arr)):
    # print('i in for loop',i)
    sex_fits = fits.open(sex_arr[i])
    sex_dat = sex_fits[0].data

    seg_fits = fits.open(seg_arr[i])
    seg_dat = seg_fits[0].data

    masking = np.where(seg_dat > 0, 0, sex_dat)

    # ============== 03. normalize ========== #
    numrow = len(masking)
    numcol = len(masking[0])

    cenrow_ind = int(numrow // 2)
    cencol_ind = int(numcol // 2)

    norm_const = masking[cenrow_ind, cencol_ind]

    if norm_const == 0:
        masking = masking + 1
        norm_image = masking
        hdu = fits.PrimaryHDU(norm_image)
        hdul = fits.HDUList([hdu])
        hdul.writeto(str(i) + '_norm.fits', overwrite=True)
        sex_fits.close()
        seg_fits.close()


    else:
        norm_image = masking / norm_const

        hdu = fits.PrimaryHDU(norm_image)
        hdul = fits.HDUList([hdu])
        hdul.writeto(str(i) + '_norm.fits', overwrite=True)
        sex_fits.close()
        seg_fits.close()

    # ---------- 02. makes data center pixel value 1 ------------ #
    # vallist = np.ravel(image)
    # uni_vallist = np.unique(vallist)
    # print(uni_vallist)


ref = fits.open('0_norm.fits')
image = ref[0].data

for i in range(1,23):
    # ---------- 01. data open ------------ #
    # print(image.shape)
    norm_fits = fits.open(str(i) + '_norm.fits')
    norm_dat = norm_fits[0].data
    image = np.dstack((norm_dat, image))


flat = np.median(image, axis=2)
plt.imshow(flat)
plt.title('fio_flat')
plt.show()
print(flat)
print(flat.shape)
hdu_flat = fits.PrimaryHDU(flat)
hdul = fits.HDUList([hdu_flat])
hdul.writeto('fio_flat.fits',overwrite=True)



