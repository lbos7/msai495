from mp7.ImageTracking import ImageTracking

image_folder = 'mp7/image_girl'
output_folder = 'mp7/tracking_output'

# Options for method: 'ssd', 'cc', 'ncc'
positions = ImageTracking(
    image_folder,
    output_folder,
    method='ncc',
    search_radius=25,
    save_video=True,
    video_name='mp7/girl_head_tracking.mp4'
)
