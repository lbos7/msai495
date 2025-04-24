import cv2
import numpy as np
from SkinDetection import SkinDetection

gun1 = cv2.imread('mp4/images/gun1.bmp')
joy1 = cv2.imread('mp4/images/joy1.bmp')
pointer1 = cv2.imread('mp4/images/pointer1.bmp')

hist_data = np.load('mp4/skin_hist.npz')

gun1_detected = SkinDetection(gun1, hist_data)
joy1_detected = SkinDetection(joy1, hist_data)
pointer1_detected = SkinDetection(pointer1, hist_data)

cv2.imshow('gun1.bmp', gun1)
cv2.imshow('joy1.bmp', joy1)
cv2.imshow('pointer1.bmp', pointer1)
cv2.imshow('SkinDetection on gun1.bmp', gun1_detected)
cv2.imshow('SkinDetection on joy1.bmp', joy1_detected)
cv2.imshow('SkinDetection on pointer1.bmp', pointer1_detected)
cv2.waitKey(0)
cv2.destroyAllWindows()