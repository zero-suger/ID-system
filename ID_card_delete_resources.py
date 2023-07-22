import os
import time

# Define the filenames to be deleted
image_files = ['captured_image_result.jpg','cropped_test01.jpg', 'id_img.jpg']
npy_files = ['test01.npy']


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
