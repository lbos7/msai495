import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from pathlib import Path

# Set up images folder
images_folder_path = 'mp4/images/'
images_folder = Path(images_folder_path)

# Create a subdirectory for cropped images if it doesn't exist
cropped_images_folder = images_folder / 'cropped_images'
cropped_images_folder.mkdir(exist_ok=True)  # Will not raise error if the folder already exists

def onselect(eclick, erelease):
    '''
    Applies the histogram equalization to a given image.
    
    Parameters:
        eclick (MouseEvent): The mouse event when the starting point of the rectangle is selected.
        erelease (MouseEvent): The mouse event when the ending point of the rectangle is released.
        
    '''

    # Get the corner coordinates of the selection
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)

    # Ensure coordinates are in correct order
    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    # Crop the image and add it to the list of cropped images
    cropped = img[ymin:ymax, xmin:xmax]
    cropped_images.append(cropped)

# Loop through images to obtained cropped images
for f in images_folder.iterdir():
    if f.is_file():

        # Read image
        img = plt.imread(images_folder_path + f.name)

        # Set up empty list of cropped images
        cropped_images = []

        # Set up plot
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(f.name)

        # Used for selecting rectangle to crop images
        rect_selector = RectangleSelector(ax, onselect, useblit=True,
                                         button=[1], minspanx=5, minspany=5, spancoords='pixels',
                                         interactive=True)
        plt.show()

        # Save cropped images in the 'cropped_images' folder
        for idx, cropped in enumerate(cropped_images):
            cropped_bgr = (cropped[:, :, :3] * 255).astype('uint8') if cropped.dtype == float else cropped
            
            # Save each cropped image in the cropped_images folder
            save_path = cropped_images_folder / f"{f.stem}_cropped_{idx+1}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(cropped_bgr, cv2.COLOR_RGB2BGR))
            print(f"Saved: {save_path}")
