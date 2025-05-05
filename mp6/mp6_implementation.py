import cv2
import numpy as np
from HoughTransform import HoughTransform
from CannyEdgeDetection import CannyEdgeDetection

input = cv2.imread('mp6/images/input.bmp')
test = cv2.imread('mp6/images/test.bmp')
test2 = cv2.imread('mp6/images/test2.bmp')

input_lines = HoughTransform(input)
test_lines = HoughTransform(test, percentageOfNonEdge=.94)
test2_lines = HoughTransform(test2, percentageOfNonEdge=.94)

# joy1_smoothed = GaussSmoothing(joy1, 5, 1)
# pointer1_smoothed = GaussSmoothing(pointer1, 5, 1)
# lena_smoothed = GaussSmoothing(lena, 5, 1)
# test1_smoothed = GaussSmoothing(test1, 5, 1)

# joy1_edges = CannyEdgeDetection(joy1, N=5, sigma=2, percentageOfNonEdge=.8)
# pointer1_edges = CannyEdgeDetection(pointer1, N=5, sigma=2, percentageOfNonEdge=.8)
# lena_edges = CannyEdgeDetection(lena, N=5, sigma=1, percentageOfNonEdge=.8)
# test1_edges = CannyEdgeDetection(test1, N=5, sigma=2, percentageOfNonEdge=.8)
# test_edges = CannyEdgeDetection(test, N=5, sigma=2, percentageOfNonEdge=.95)


cv2.imshow('input.bmp', input)
cv2.imshow('test.bmp', test)
cv2.imshow('test2.bmp', test2)
cv2.imshow('Lines of input.bmp', input_lines)
cv2.imshow('Lines of test.bmp', test_lines)
cv2.imshow('Lines of test2.bmp', test2_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()