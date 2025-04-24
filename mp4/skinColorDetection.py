import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

images_folder_path = 'mp4/images/'
image_filenames = ['gun1.bmp', 'joy1.bmp', 'pointer1.bmp']
img = plt.imread('mp4/images/gun1.bmp')


def onselect(eclick, erelease):
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    cropped = img[y1:y2, x1:x2]
    plt.figure()
    plt.imshow(cropped)
    plt.title("Cropped")
    plt.show()

fig, ax = plt.subplots()
ax.imshow(img)
rect_selector = RectangleSelector(ax, onselect, drawtype='box', useblit=True,
                                   button=[1], minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
plt.show()