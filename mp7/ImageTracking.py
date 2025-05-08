import cv2
import numpy as np
import os

def compute_similarity(patch, template, method='ncc'):
    if method == 'ssd':
        return -np.sum((patch - template) ** 2)
    elif method == 'cc':
        return np.sum(patch * template)
    elif method == 'ncc':
        patch_mean = np.mean(patch)
        template_mean = np.mean(template)
        num = np.sum((patch - patch_mean) * (template - template_mean))
        denom = np.sqrt(np.sum((patch - patch_mean)**2) * np.sum((template - template_mean)**2))
        return num / denom if denom != 0 else 0

def template_match_track(
    image_folder,
    output_folder,
    method='ncc',
    search_radius=20,
    save_video=False,
    video_name='mp7/output_tracking.mp4'
):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')])
    os.makedirs(output_folder, exist_ok=True)

    first_frame = cv2.imread(os.path.join(image_folder, image_files[0]))
    bbox = cv2.selectROI("Select Target", first_frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = map(int, bbox)
    cv2.destroyAllWindows()
    template = cv2.cvtColor(first_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

    positions = [(x, y, w, h)]

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = first_frame.shape[:2]
        out = cv2.VideoWriter(video_name, fourcc, 30, (width, height))

    total_frames = len(image_files)
    for i, fname in enumerate(image_files):
        print(f"Processing frame {i+1}/{total_frames}")
        img_path = os.path.join(image_folder, fname)
        frame = cv2.imread(img_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if i == 0:
            frame_draw = frame.copy()
            cv2.rectangle(frame_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_folder, fname), frame_draw)
            if save_video:
                out.write(frame_draw)
            continue

        prev_x, prev_y, _, _ = positions[-1]
        best_score = -np.inf
        best_pos = (prev_x, prev_y)

        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                new_x = prev_x + dx
                new_y = prev_y + dy

                if (new_x < 0 or new_y < 0 or
                    new_x + w > gray.shape[1] or new_y + h > gray.shape[0]):
                    continue

                patch = gray[new_y:new_y + h, new_x:new_x + w]
                score = compute_similarity(patch.astype(np.float32), template.astype(np.float32), method)

                if score > best_score:
                    best_score = score
                    best_pos = (new_x, new_y)

        x, y = best_pos
        positions.append((x, y, w, h))
        frame_draw = frame.copy()
        cv2.rectangle(frame_draw, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_folder, fname), frame_draw)
        if save_video:
            out.write(frame_draw)

    if save_video:
        out.release()

    print("Tracking complete.")
    return positions
