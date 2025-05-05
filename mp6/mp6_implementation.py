import cv2
from HoughTransform import HoughTransform

input = cv2.imread('mp6/images/input.bmp')
test = cv2.imread('mp6/images/test.bmp')
test2 = cv2.imread('mp6/images/test2.bmp')

input_lines = HoughTransform(input, param_quant=180, max_filt_N=15, peaks_thresh_mul=.61)
test_lines = HoughTransform(test, percentageOfNonEdge=.94, param_quant=180, max_filt_N=15, peaks_thresh_mul=.21)
test2_lines = HoughTransform(test2, percentageOfNonEdge=.94, param_quant=180, max_filt_N=15, peaks_thresh_mul=.22)

cv2.imshow('input.bmp', input)
cv2.imshow('test.bmp', test)
cv2.imshow('test2.bmp', test2)
cv2.imshow('Line Detection on input.bmp', input_lines)
cv2.imshow('Line Detection on test.bmp', test_lines)
cv2.imshow('Line Detection on test2.bmp', test2_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()