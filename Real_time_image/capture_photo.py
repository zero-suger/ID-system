import cv2
import os
import time

def capture_image(folder_path):
    # Open the webcam
    camera_ip = '192.168.0.22'
    username = 'root'
    password = 'kiicti'

    # Construct the RTSP URL
    rtsp_url = f'rtsp://{username}:{password}@{camera_ip}/axis-media/media.amp'
    cap = cv2.VideoCapture(rtsp_url)

    # Set the frame width and height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create a window for the GUI
    cv2.namedWindow("Capture Image")

    # Calculate the center coordinates of the frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_center_x = frame_width // 2
    frame_center_y = frame_height // 2

    # Calculate the size of the rectangular frame
    frame_size = min(frame_width, frame_height) // 2
    frame_x1 = frame_center_x - frame_size
    frame_y1 = frame_center_y - frame_size
    frame_x2 = frame_center_x + frame_size
    frame_y2 = frame_center_y + frame_size

    # Start the countdown timer
    countdown = 30
    start_time = time.time()

    while countdown > 0:
        # Read a frame from the webcam
        ret, frame = cap.read()

        # Draw a frame around the face
        cv2.rectangle(frame, (frame_x1, frame_y1), (frame_x2, frame_y2), (0, 255, 0), 2)

        # Display the countdown and frame
        cv2.putText(frame, f"Capture in {countdown} seconds", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Capture Image", frame)

        # Calculate the remaining time
        elapsed_time = time.time() - start_time
        countdown = 5 - int(elapsed_time)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Capture the image
    ret, frame = cap.read()

    # Crop the image using the rectangular frame coordinates
    cropped_frame = frame[frame_y1:frame_y2, frame_x1:frame_x2]

    # Specify the folder path to save the image
    save_folder_path = "/home/suger01/Desktop/ID_SYSTEM/Real_time_image/captured_image"  # Update with the desired folder path

    # Specify the file name
    file_name = "captured_image.jpg"

    # Combine the folder path and file name
    file_path = os.path.join(save_folder_path, file_name)

    # Save the cropped image to the specified file path
    cv2.imwrite(file_path, cropped_frame)

    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

# Specify the folder path where you want to save the image
folder_path = "/home/suger01/Desktop/ID_SYSTEM"

# Call the function to capture the image and specify the folder path
capture_image(folder_path)
