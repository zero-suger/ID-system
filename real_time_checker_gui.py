import tkinter as tk
from tkinter import messagebox
import subprocess

def take_picture():
    # Check the click count
    if take_picture.count == 1:
        # Show information in a pop-up window
        messagebox.showinfo("Information","This is './Real_time_image/capture_photo.py' file which takes your picture and save it to './Real_time_image/capture_image' folder to compare your Real-time picture with ID picture.")
    elif take_picture.count == 2:
        # Run the specified Python file
        subprocess.run(['python', './Real_time_image/capture_photo.py'])

    # Increment the click count
    take_picture.count += 1

def check_matching():
    # Check the click count
    if check_matching.count == 1:
        # Show information in a pop-up window
        messagebox.showinfo("Information", "When you click 'Check Matching', the captured picture will be compared with the ID picture. (After the 'Take Picture' step) -> Core_detector.py -> Core_recognizer.py -> angular_distance.py -> final_similarity_cv.py -> subprocess_real_time_files.py.")
    elif check_matching.count == 2:
        # Run the specified Python file
        subprocess.run(['python', './subprocess_real_time_files.py'])

    # Increment the click count
    check_matching.count += 1

def check_spoofing():
    # Run the specified Python file and capture the output
    result = subprocess.run(['python', './Silent-Face-Anti-Spoofing/test.py'], capture_output=True, text=True)
    output = result.stdout

    # Display the output in the GUI
    messagebox.showinfo("Spoofing Result", output)

    # Check if the output contains "Fake Face"
    if "Fake Face" in output:
        # Display warning message
        messagebox.showwarning("Warning", "You are using a fake face. We can't allow you to use the 'Check Matching' button. Try again with an original face.")

        # After 5 seconds, close the GUI
        window.after(100, exit_app)

def exit_app():
    window.destroy()

def delete_files():
    # Check the click count
    if delete_files.count == 1:
        # Show information in a pop-up window
        messagebox.showinfo("Information", "When you click 'Delete Real Time Resources', you will delete all .jpg and .npy files related to the Real time GUI in 3 seconds. It is better to run every time at the end.")
    elif delete_files.count == 2:
        # Run the specified Python file
        subprocess.run(['python', './delete_files_real_time.py'])
    
    delete_files.count += 1
   

# Set the initial click counts
take_picture.count = 1
check_matching.count = 1
delete_files.count = 1
# Create the main window
window = tk.Tk()
window.title("Real Time Face Similarity Detect")
window.geometry("350x600")

# Resize the logo image to 100x100 pixels
logo_image = tk.PhotoImage(file="face_recognition_icon.png")
resized_logo = logo_image.subsample(8)  # Change the number to adjust the size
logo_label = tk.Label(window, image=resized_logo)
logo_label.pack()

# Create a label for displaying information
info_label = tk.Label(window, text="", font=("Arial", 12))
info_label.pack(pady=20)

# Create a button for taking a picture
picture_button = tk.Button(window, text="Take Picture", font=("Arial", 14, "bold"), width=25, height=2, command=take_picture)
picture_button.pack(pady=10)

# Create a button for checking spoofing
spoofing_button = tk.Button(window, text="Check Spoofing", font=("Arial", 14, "bold"), width=25, height=2, command=check_spoofing)
spoofing_button.pack(pady=10)

# Create a button for checking matching
matching_button = tk.Button(window, text="Check Matching", font=("Arial", 14, "bold"), width=25, height=2, command=check_matching)
matching_button.pack(pady=10)

# Create a button for deleting real-time sources
delete_button = tk.Button(window, text="Delete Real Time Sources", font=("Arial", 14, "bold"), width=25, height=2, command=delete_files)
delete_button.pack(pady=10)



# Create an exit button
exit_button = tk.Button(window, text="Exit", font=("Arial", 14, "bold"), width=17, height=2, command=exit_app)
exit_button.pack(pady=55)

# Create a label for copyright information
copyright_label = tk.Label(window, text="Urinov Azizbek 2023 ©", font=("Arial", 12, "bold"), fg="red")
copyright_label.pack(side=tk.BOTTOM)



# Run the main event loop
window.mainloop()
