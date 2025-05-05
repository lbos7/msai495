import cv2
import numpy as np
from CannyEdgeDetection import GaussSmoothing, CannyEdgeDetection

joy1 = cv2.imread('mp5/images/joy1.bmp')
pointer1 = cv2.imread('mp5/images/pointer1.bmp')
lena = cv2.imread('mp5/images/lena.bmp')
test1 = cv2.imread('mp6/images/test.bmp')


joy1_smoothed = GaussSmoothing(joy1, 5, 1)
pointer1_smoothed = GaussSmoothing(pointer1, 5, 1)
lena_smoothed = GaussSmoothing(lena, 5, 1)
test1_smoothed = GaussSmoothing(test1, 5, 1)

joy1_edges = CannyEdgeDetection(joy1, N=5, sigma=2, percentageOfNonEdge=.8)
pointer1_edges = CannyEdgeDetection(pointer1, N=5, sigma=2, percentageOfNonEdge=.8)
lena_edges = CannyEdgeDetection(lena, N=5, sigma=1, percentageOfNonEdge=.8)
test1_edges = CannyEdgeDetection(test1, N=5, sigma=2, percentageOfNonEdge=.8)

cv2.imshow('joy1.bmp', joy1)
cv2.imshow('pointer1.bmp', pointer1)
cv2.imshow('lena.bmp', lena)
cv2.imshow('test1.bmp', test1)
cv2.imshow('Smoothed joy1.bmp', joy1_smoothed)
cv2.imshow('Smoothed pointer1.bmp', pointer1_smoothed)
cv2.imshow('Smoothed lena.bmp', lena_smoothed)
cv2.imshow('Smoothed test1.bmp', test1_smoothed)
cv2.imshow('Edges of joy1.bmp', joy1_edges)
cv2.imshow('Edges of pointer1.bmp', pointer1_edges)
cv2.imshow('Edges of lena.bmp', lena_edges)
cv2.imshow('Edges of test1.bmp', test1_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()