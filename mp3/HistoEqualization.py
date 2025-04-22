import cv2
import numpy as np
import matplotlib.pyplot as plt

def HistoEqualization(img):

    # Converting image to grayscale and determining dimensions
    img_gray = np.copy(img)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    rows,cols = img_gray.shape

    hist = np.histogram(img_gray.flatten(), bins=256, density=True)
    cdf = np.cumsum(hist)

    L2_arr = cdf*255
    
    for i in range(rows):
        for j in range(cols):
            img_gray[i, j] = int(L2_arr[img_gray[i, j]])

    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    # plt.figure(1)
    # plt.plot(cdf * hist.max(), color='b')
    # plt.hist(img_gray.flatten(), bins=256, density=True, color='r')
    # # plt.stairs(hist, bins)
    # plt.legend(['cdf (scaled)', 'histogram'])
    # plt.title("moon.bmp histogram")
    # plt.show()