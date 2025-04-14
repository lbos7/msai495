import cv2
import numpy as np
from CCL import CCL

test_img = cv2.imread('mp1/images/test.bmp')
face_img = cv2.imread('mp1/images/face.bmp')

test_img_labeled,test_region_num = CCL(test_img)
face_img_labeled,face_region_num = CCL(face_img)

print("Number of regions in test.bmp: " + str(test_region_num))
print("Number of regions in face.bmp: " + str(face_region_num))

cv2.imshow('test.bmp', test_img)
cv2.imshow('labeled test.bmp', test_img_labeled)
cv2.imshow('face.bmp', face_img)
cv2.imshow('labeled face.bmp', face_img_labeled)
cv2.waitKey(0)
cv2.destroyAllWindows()