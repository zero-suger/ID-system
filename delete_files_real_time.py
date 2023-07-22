import os
import time

# Define the filenames to be deleted
image_files = ['captured_image_result.jpg','cropped_test02.jpg', 'real_time_img.jpg']
npy_files = ['test02.npy', 'similarity_score.npy']


delete_after_seconds = 3  # 3 seconds

# Wait for 1 minutes
time.sleep(delete_after_seconds)

# Delete the image files
for file in image_files:
    if os.path.exists(file):
        os.remove(file)

# Delete the NPY files
for file in npy_files:
    if os.path.exists(file):
        os.remove(file)
