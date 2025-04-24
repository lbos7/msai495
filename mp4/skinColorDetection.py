import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from pathlib import Path

images_folder_path = 'mp4/images/'
images_folder = Path(images_folder_path)
image_filenames = ['gun1.bmp', 'joy1.bmp', 'pointer1.bmp']
img = plt.imread('mp4/images/gun1.bmp')

def onselect(eclick, erelease):
    # Get corner coordinates
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

    # Ensure coordinates are in correct order
    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    # Crop the image
    cropped = img[ymin:ymax, xmin:xmax]

    # Show cropped image
    plt.figure()
    plt.imshow(cropped)
    plt.title("Cropped Image")
    plt.axis("off")
    plt.show()

    # Save using OpenCV (handles JPG/PNG well)
    # save_path = 'cropped_output.jpg'
    global cropped_bgr
    cropped_bgr = (cropped[:, :, :3] * 255).astype('uint8') if cropped.dtype == float else cropped

fig, ax = plt.subplots()


for f in images_folder.iterdir():
    if f.is_file() and f.name[-4:] == '.bmp':
        print(f.name)
        img = plt.imread(images_folder_path + f.name)
        ax.imshow(img)
        rect_selector = RectangleSelector(ax, onselect, drawtype='box', useblit=True,
                                   button=[1], minspanx=5, minspany=5, spancoords='pixels',
                                   interactive=True)
        plt.show()
        save_path = images_folder_path + f.name[:-4] + '_cropped.png'
        cv2.imwrite(save_path, cv2.cvtColor(cropped_bgr, cv2.COLOR_RGB2BGR))