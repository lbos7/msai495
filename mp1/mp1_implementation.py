import cv2
import numpy as np
from CCL import CCL

test_img = cv2.imread('mp1/images/test.bmp')
face_img = cv2.imread('mp1/images/face.bmp')
gun_img = cv2.imread('mp1/images/gun.bmp')

filter_thresh = 224

test_img_labeled,test_region_num = CCL(test_img)
face_img_labeled,face_region_num = CCL(face_img)
gun_img_labeled,gun_region_num = CCL(gun_img)
gun_filt_img_labeled,gun_filt_region_num = CCL(gun_img, use_size_filter=True, filter_thresh=filter_thresh)

print("Number of regions in test.bmp: " + str(test_region_num))
print("Number of regions in face.bmp: " + str(face_region_num))
print("Number of regions in gun.bmp: " + str(gun_region_num))
print("Number of regions in gun.bmp with " +  str(filter_thresh) + " pixel filter: " + str(gun_filt_region_num))

cv2.imshow('test.bmp', test_img)
cv2.imshow('labeled test.bmp', test_img_labeled)
cv2.imshow('face.bmp', face_img)
cv2.imshow('labeled face.bmp', face_img_labeled)
cv2.imshow('gun.bmp', gun_img)
cv2.imshow('labeled gun.bmp', gun_img_labeled)
cv2.imshow('gun.bmp', gun_img)
cv2.imshow('labeled gun.bmp with size filter', gun_filt_img_labeled)
cv2.waitKey(0)
cv2.destroyAllWindows()