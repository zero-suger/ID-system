import cv2
import numpy as np

# Define the desired width and height for the images
image_width = 250
image_height = 250

# Load the images
image1 = cv2.imread('/home/suger01/Desktop/ID_SYSTEM/cropped_test01.jpg')
image2 = cv2.imread('/home/suger01/Desktop/ID_SYSTEM/cropped_test02.jpg')

# Resize the images to the desired width and height
image1 = cv2.resize(image1, (image_width, image_height))
image2 = cv2.resize(image2, (image_width, image_height))

# Create a blank canvas to display the images and data
canvas = np.zeros((image_height, image_width * 2, 3), dtype=np.uint8)

# Place the images on the canvas
canvas[:image_height, :image_width] = image1
canvas[:image_height, image_width:] = image2

# Load the similarity score
data = np.load('similarity_score.npy')

# Format the similarity score as a percentage
similarity_percentage = int(data * 100)

# Create the data text
data_text = f"Similarity Score: {similarity_percentage}%"

# Add the data at the top center of the canvas
text_font = cv2.FONT_HERSHEY_SIMPLEX
text_scale = 0.8
text_thickness = 2

text_size, _ = cv2.getTextSize(data_text, text_font, text_scale, text_thickness)
text_x = int((canvas.shape[1] - text_size[0]) / 2)
text_y = text_size[1] + 10  # Distance from the top of the canvas

cv2.putText(canvas, data_text, (text_x, text_y), text_font, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)

# Display the canvas
cv2.imshow('Images with Data', canvas)

# Wait for a key press to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
