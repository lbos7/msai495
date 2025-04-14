import cv2
import numpy as np
from CCL import CCL

test_img = cv2.imread('mp1/images/test.bmp')

test_img_labeled,test_region_num = CCL(test_img)

print(test_region_num)

cv2.imshow('image', test_img)

cv2.imshow('labeled image', test_img_labeled)
cv2.waitKey(0)
cv2.destroyAllWindows()