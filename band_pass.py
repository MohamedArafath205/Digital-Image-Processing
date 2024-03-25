import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('assets/cat.jpeg', 0)

if img is None:
    print("Error loading the image.")
else:
    blurred_img = cv2.GaussianBlur(img, (0, 0), 2.0)
    sharpened_img = cv2.addWeighted(img, 1.5, blurred_img, -0.5, 0)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(sharpened_img, cmap='gray')
    plt.title('Sharpened Image (Unsharp Masking)'), plt.xticks([]), plt.yticks([])
    plt.show()
