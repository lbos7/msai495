import cv2
import numpy as np
import os

def ComputeSimilarity(patch, template, method='ncc'):
    '''
    Computes the similarity between an image patch and a template using the specified method.

    Parameters:
        patch (numpy array): Grayscale image patch from the current frame.
        template (numpy array): Grayscale template from the initial frame.
        method (str): Similarity metric to use. Options are 'ssd', 'cc', or 'ncc'.

    Returns:
        float: Similarity score based on the selected method.
               Higher values indicate a better match (except for SSD where we return the negative value).
    '''
    # Case for Sum of Squared Difference
    if method == 'ssd':
        return -np.sum((patch - template) ** 2)
    
    # Case for Cross-Correlation
    elif method == 'cc':
        return np.sum(patch * template)
    
    # Case for Normalized Cross-Correlation
    elif method == 'ncc':
        patch_mean = np.mean(patch)
        template_mean = np.mean(template)
        num = np.sum((patch - patch_mean) * (template - template_mean))
        denom = np.sqrt(np.sum((patch - patch_mean)**2) * np.sum((template - template_mean)**2))
        return num / denom if denom != 0 else 0


def ImageTracking(
    image_folder,
    output_folder,
    method='ncc',
    search_radius=20,
    save_video=False,
    video_name='mp7/output_tracking.mp4'
):
    '''
    Performs template-matching based object tracking on a sequence of image frames.

    Parameters:
        image_folder (str): Path to folder containing input .jpg frames.
        output_folder (str): Path to folder for saving output frames with drawn bounding boxes.
        method (str): Similarity metric for template matching ('ssd', 'cc', or 'ncc').
        search_radius (int): Number of pixels to search in each direction from previous target location.
        save_video (bool): If True, saves the tracking result as a video.
        video_name (str): File name for the output video if save_video is True.

    Returns:
        positions (list of tuples): A list of (x, y, w, h) tuples representing the target bounding box
                                    in each frame.
    '''

    # Sorting image files to make sure they're in order and creating output directory if necessary
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    os.makedirs(output_folder, exist_ok=True)

    # Selecting intial frame as template
    first_frame = cv2.imread(os.path.join(image_folder, image_files[0]))
    bbox = cv2.selectROI("Select Target", first_frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = map(int, bbox)
    cv2.destroyAllWindows()
    template = cv2.cvtColor(first_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

    # Setting up postitions list
    positions = [(x, y, w, h)]

    # Option to save video
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = first_frame.shape[:2]
        out = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

    # Looping through each image to track the template
    total_frames = len(image_files)
    for i, fname in enumerate(image_files):

        # Output message
        print(f"Processing frame {i+1}/{total_frames}")

        # Retrieving image and converting to grayscale
        img_path = os.path.join(image_folder, fname)
        frame = cv2.imread(img_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # For the first frame draw rectangle and save image (add to video if necessary)
        if i == 0:
            frame_draw = frame.copy()
            cv2.rectangle(frame_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_folder, fname), frame_draw)
            if save_video:
                out.write(frame_draw)
            continue

        # Setting up variables for tracking
        prev_x, prev_y, _, _ = positions[-1]
        best_score = -np.inf
        best_pos = (prev_x, prev_y)

        # Search for the best match in the defined radius around the previous position
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                new_x = prev_x + dx
                new_y = prev_y + dy

                if (new_x < 0 or new_y < 0 or
                    new_x + w > gray.shape[1] or new_y + h > gray.shape[0]):
                    continue

                patch = gray[new_y:new_y + h, new_x:new_x + w]
                score = ComputeSimilarity(patch.astype(np.float32), template.astype(np.float32), method)

                if score > best_score:
                    best_score = score
                    best_pos = (new_x, new_y)

        # Draw the best matching bounding box and save the result
        x, y = best_pos
        positions.append((x, y, w, h))
        frame_draw = frame.copy()
        cv2.rectangle(frame_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_folder, fname), frame_draw)
        if save_video:
            out.write(frame_draw)

    # Finalize and save the video if needed
    if save_video:
        out.release()

    return positions
