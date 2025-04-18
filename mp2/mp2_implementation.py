import cv2
import numpy as np
from MorphologicalOperators import Dilation, Erosion, Opening, Closing, Boundary

gun_img = cv2.imread('mp2/images/gun.bmp')
palm_img = cv2.imread('mp2/images/palm.bmp')

se = (7, 7)
se2 = (3, 3)
gun_img_dilated = Dilation(gun_img, se)
gun_img_eroded = Erosion(gun_img, se)
gun_img_opened = Opening(gun_img, se)
gun_img_closed = Closing(gun_img, se)
gun_img_boundary = Boundary(gun_img_closed, se2)

palm_img_dilated = Dilation(palm_img, se)
palm_img_eroded = Erosion(palm_img, se)
palm_img_opened = Opening(palm_img, se)
palm_img_closed = Closing(palm_img, se)
palm_img_closed2 = Erosion(palm_img_dilated, se2)
palm_img_boundary = Boundary(palm_img_closed2, se2)

cv2.imshow('gun.bmp', gun_img)
cv2.imshow('dilated gun.bmp', gun_img_dilated)
cv2.imshow('eroded gun.bmp', gun_img_eroded)
cv2.imshow('opened gun.bmp', gun_img_opened)
cv2.imshow('closed gun.bmp', gun_img_closed)
cv2.imshow('boundary of gun.bmp', gun_img_boundary)
cv2.imshow('palm.bmp', palm_img)
cv2.imshow('dilated palm.bmp', palm_img_dilated)
cv2.imshow('eroded palm.bmp', palm_img_eroded)
cv2.imshow('opened palm.bmp', palm_img_opened)
cv2.imshow('closed palm.bmp', palm_img_closed)
cv2.imshow('boundary of palm.bmp', palm_img_boundary)
cv2.waitKey(0)
cv2.destroyAllWindows()