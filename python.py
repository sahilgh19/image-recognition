import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import time
from datetime import datetime  # Import the datetime module for timestamps

# Load Haar Cascade for face and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Initialize the webcam with a specific resolution
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width of the frame (640 pixels)
cap.set(4, 480)  # Height of the frame (480 pixels)

# Create the main window
root = tk.Tk()
root.title("Real-Time Face Detection")
root.geometry("700x600")

# Set up the canvas to display the webcam frame
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Set up the font for overlay text
font = cv2.FONT_HERSHEY_SIMPLEX

# Variables for keeping track of stats
face_count = 0
start_time = time.time()

# Function to determine if smile is genuine or fake based on its size
def smile_probability(sx, sy, sw, sh):
    # A simple rule-based approach to estimate the "genuineness" of the smile
    # Larger smile dimensions are considered genuine (this is a simplistic assumption)
    smile_area = sw * sh
    if smile_area > 1000:  # Threshold for a larger smile
        return "Genuine", 90  # High probability for genuine smile
    elif smile_area > 500:  # Medium-sized smile
        return "Semi-Genuine", 60  # Moderate probability
    else:
        return "Fake", 30  # Lower probability for fake smile

def update_frame():
    global face_count
    # Read the current frame from the webcam
    ret, img = cap.read()
    if not ret:
        messagebox.showerror("Error", "Failed to grab frame.")
        root.quit()

    # Convert the image to grayscale for both face and smile detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    face_count = len(faces)

    # Detect smiles inside detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw face rectangle

        # Region of interest (ROI) for smile detection within each face
        roi_gray = gray[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(img, (x + sx, y + sy), (x + sx + sw, y + sy + sh), (0, 255, 0), 2)  # Draw smile rectangle

            # Get smile classification and probability
            smile_type, prob = smile_probability(sx, sy, sw, sh)
            # Display smile type and probability
            cv2.putText(img, f"{smile_type} ({prob}%)", (x + sx, y + sy - 10), font, 0.6, (255, 255, 255), 2)

    # Add overlay text for face count and instructions
    cv2.putText(img, f"Faces: {face_count}", (10, 30), font, 0.9, (0, 255, 0), 2)
    cv2.putText(img, "Press 'Stop' to Exit", (10, 460), font, 0.6, (0, 0, 255), 2)

    # Get the current time and format it
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(img, current_time, (10, 470), font, 0.6, (255, 255, 255), 1)

    # Convert the image from OpenCV format (BGR) to PIL format (RGB)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # Convert PIL image to Tkinter-compatible format
    img_tk = ImageTk.PhotoImage(image=img_pil)

    # Update the canvas with the new image
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.img_tk = img_tk  # Keep a reference to the image to prevent garbage collection

    # Continue updating the frame every 10ms
    root.after(10, update_frame)

def stop_video():
    # Release the webcam and close the window
    cap.release()
    root.quit()

def capture_image():
    # Capture the current frame and save it as an image
    ret, img = cap.read()
    if ret:
        filename = f"captured_face_{int(time.time())}.jpg"
        cv2.imwrite(filename, img)
        messagebox.showinfo("Image Saved", f"Image saved as {filename}")

# Add buttons for stop and capture with background and text color
stop_button = tk.Button(root, text="Stop", width=15, command=stop_video, bg="red", fg="white", font=("Arial", 12, "bold"))
stop_button.pack(pady=10)

capture_button = tk.Button(root, text="Capture Image", width=15, command=capture_image, bg="green", fg="white", font=("Arial", 12, "bold"))
capture_button.pack(pady=10)

# Start the video stream update loop
update_frame()

# Run the Tkinter main loop
root.mainloop()

# Release the webcam and close all windows after the loop ends
cap.release()
cv2.destroyAllWindows()
