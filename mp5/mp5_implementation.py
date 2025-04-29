import cv2
import numpy as np
from CannyEdgeDetection import GaussSmoothing

joy1 = cv2.imread('mp5/images/joy1.bmp')
pointer1 = cv2.imread('mp5/images/pointer1.bmp')
lena = cv2.imread('mp5/images/lena.bmp')
test1 = cv2.imread('mp5/images/test1.bmp')


joy1_smoothed = GaussSmoothing(joy1, 5, 1)
pointer1_smoothed = GaussSmoothing(pointer1, 5, 1)
lena_smoothed = GaussSmoothing(lena, 5, 1)
test1_smoothed = GaussSmoothing(test1, 5, 1)

cv2.imshow('joy1.bmp', joy1)
cv2.imshow('pointer1.bmp', pointer1)
cv2.imshow('lena.bmp', lena)
cv2.imshow('test1.bmp', test1)
cv2.imshow('Smoothed joy1.bmp', joy1_smoothed)
cv2.imshow('Smoothed pointer1.bmp', pointer1_smoothed)
cv2.imshow('Smoothed lena.bmp', lena_smoothed)
cv2.imshow('Smoothed test1.bmp', test1_smoothed)
cv2.waitKey(0)
cv2.destroyAllWindows()