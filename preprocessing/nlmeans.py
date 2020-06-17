import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import img_as_uint
from skimage import io

img = io.imread('../testvectors/ref.tif',as_gray=True)
#img = img.astype(np.int8)


print(img.dtype)

denoise = cv2.medianBlur(img,3)

denoise2 = cv2.medianBlur(img,3)
# fast algorithm
denoise_fast = cv2.bilateralFilter(img,9,75,75)

# fast algorithm, sigma provided
denoise2_fast = denoise

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 6),
                       sharex=True, sharey=True)

ax[0, 0].imshow(img)
ax[0, 0].axis('off')
ax[0, 0].set_title('noisy')
ax[0, 1].imshow(denoise)
ax[0, 1].axis('off')
ax[0, 1].set_title('non-local means\n(slow)')
ax[0, 2].imshow(denoise2)
ax[0, 2].axis('off')
ax[0, 2].set_title('non-local means\n(slow, using $\sigma_{est}$)')
ax[1, 0].imshow(img)
ax[1, 0].axis('off')
ax[1, 0].set_title('original\n(noise free)')
ax[1, 1].imshow(denoise_fast)
ax[1, 1].axis('off')
ax[1, 1].set_title('non-local means\n(fast)')
ax[1, 2].imshow(denoise2_fast)
ax[1, 2].axis('off')
ax[1, 2].set_title('non-local means\n(fast, using $\sigma_{est}$)')

fig.tight_layout()

plt.show()
