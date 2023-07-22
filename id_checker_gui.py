import tkinter as tk
import subprocess
import PIL.Image, PIL.ImageTk
from tkinter import messagebox
def start_scanning():
    # Run the specified Python file
    subprocess.run(['python', 'subprocess_files.py'])




# Create the main window
window = tk.Tk()
window.title("ID VERIFICATION")
window.geometry("350x600")

# Load the logo image
logo = PIL.Image.open('id_recognition_icon.png')
logo = logo.resize((100, 100), PIL.Image.Resampling.LANCZOS)
logo_tk = PIL.ImageTk.PhotoImage(logo)

# Add a logo label to the top of the window
logo_label = tk.Label(window, image=logo_tk)
logo_label.pack(side="top", padx=10, pady=10)

def delete_files():
    # Check the click count
    if delete_files.count == 1:
        # Show information in a pop-up window
        messagebox.showinfo("Information", "When you click 'Delete Real Time Resources', you will delete all .jpg and .npy files related to the Real time GUI in 3 seconds. It is better to run every time at the end.")
    elif delete_files.count == 2:
        # Run the specified Python file
        subprocess.run(['python', './ID_card_delete_resources.py'])
    
    delete_files.count += 1
   

# Set the initial click counts

delete_files.count = 1

def exit_app():
    window.destroy()
    
    
# Create a button for deleting real-time sources
delete_button = tk.Button(window, text="Delete Real Time Sources", font=("Arial", 14, "bold"), width=25, height=2, command=delete_files)
delete_button.pack(pady=10)

# Create a button for starting scanning
scan_button = tk.Button(window, text="Start Scanning", font=("Arial", 14, "bold"), width=25, height=2, command=start_scanning)
scan_button.pack(pady=20)

# Add a text label under the start scanning button
scan_button_label = tk.Label(window, text="Before pressing the 'Start Scanning' button, change 'image_path' in line 93 to your ID card picture PATH.\n\nAlso check you don't have any current .npy file or images in working directory.\n\n Lastly, to system verify your picture is in DB, you should add embeddings in database.", font=("Arial", 10, "bold"), wraplength=280)
scan_button_label.pack(pady=10)

# Stick the exit button to the bottom of the window
# Create an exit button
exit_button = tk.Button(window, text="Exit", font=("Arial", 14, "bold"), width=17, height=2, command=exit_app)
exit_button.pack(pady=55)

# Create a label for copyright information
copyright_label = tk.Label(window, text="Urinov Azizbek 2023 ©", font=("Arial", 12, "bold"), fg="red")
copyright_label.pack(side=tk.BOTTOM)
# Run the main event loop
window.mainloop()
