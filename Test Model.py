import numpy as np
import tensorflow as tf
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model('eye_detection_model.h5')

# Specify the directory where your test images are located
image_directory = r'D:\Learning & Practice Vault\DS Practise notebook\Projects\Museum Artifact Tracking System\test'

# List all the image files in the specified directory
image_files = [f for f in os.listdir(image_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

# Initialize counters
open_eye_count = 0
closed_eye_count = 0

# Process each image
for image_file in image_files:
    # Create the full path to the image file
    image_path = os.path.join(image_directory, image_file)

    # Load and preprocess the image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is not None:
        # Resize the image to the desired dimensions (64x64)
        image = cv2.resize(image, (64, 64))

        # Normalize the image data (if not already in the range [0, 1])
        image = image / 255.0

        # Make a prediction
        prediction = model.predict(np.array([image]))

        # Assuming your model is for binary classification (open or closed eye)
        if prediction[0][0] >= 0.5:
            print(f"Image: {image_file} - Closed  Eye")
            closed_eye_count += 1
        else:
            print(f"Image: {image_file} - Open Eye")

            open_eye_count += 1
    else:
        print(f"Image: {image_file} - Error: Failed to load the image.")

# Print the counts
print(f"Open Eye Count: {open_eye_count}")
print(f"Closed Eye Count: {closed_eye_count}")
