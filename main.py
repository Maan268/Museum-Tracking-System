import os
import cv2
import numpy as np

# Specify the directory where your dataset is located
dataset_dir = r"D:\Learning & Practice Vault\DS Practise notebook\Data Set\d\Eye Data Set\data"

# Create empty lists to store the preprocessed data and labels
X_train = []
X_test = []
y_train = []  # For storing labels corresponding to X_train
y_test = []   # For storing labels corresponding to X_test

# Define the subfolders
subfolders = ["train", "test"]

# Loop through the subfolders ("train" and "test")
for subfolder in subfolders:
    subfolder_path = os.path.join(dataset_dir, subfolder)

    # Create empty lists for each subfolder
    data = []

    # Define the classes within each subfolder
    classes = ["open eyes", "close eyes"]

    # Loop through the classes ("open eye" and "close eye") within the subfolder
    for class_index, class_name in enumerate(classes):
        class_path = os.path.join(subfolder_path, class_name)

        # Loop through the images in each class
        for filename in os.listdir(class_path):
            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)

            if image is not None:
                # Perform image preprocessing (e.g., resize, normalize)
                image = cv2.resize(image, (64, 64))
                image = image / 255.0

                # Append the preprocessed image to the data list
                data.append(image)

                # Append the corresponding label (class_index) to the labels list
                if subfolder == "train":
                    y_train.append(class_index)
                elif subfolder == "test":
                    y_test.append(class_index)

    # Depending on the subfolder, append data to X_train or X_test
    if subfolder == "train":
        X_train = np.array(data)
    elif subfolder == "test":
        X_test = np.array(data)

# Now you have X_train, X_test, y_train, and y_test containing preprocessed data and labels
# You can save these arrays to files if needed
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
