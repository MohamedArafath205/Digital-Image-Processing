import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('assets/cat.jpeg', 0)

if img is None:
    print("Error loading the image.")
else:
    mean = 0
    sigma = 15  
    rows, cols = img.shape
    noise = np.random.normal(mean, sigma, (rows, cols))

    noisy_img = img.astype(np.float32) + noise.astype(np.float32)

    noisy_img = np.clip(noisy_img, 0, 255)
    noisy_img = noisy_img.astype(np.uint8)

    f = np.fft.fft2(noisy_img)
    fshift = np.fft.fftshift(f)

    crow, ccol = rows // 2, cols // 2
    fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(noisy_img, cmap='gray')
    plt.title('Noisy Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img_back, cmap='gray')
    plt.title('Band Reject Filtered Image'), plt.xticks([]), plt.yticks([])
    plt.show()
