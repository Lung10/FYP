import numpy as np
import os
from keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('/Users/walau/OneDrive/Desktop/model/cnn/age_gender_model_50epochs.h5')

# Folder path containing the test images
folder_path = '/Users/walau/OneDrive/Desktop/model/test_dataset'  # Replace with the actual folder path containing the test images

# Loop through each image in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.png', '.bmp')):
        # Load and preprocess the test image
        image_path = os.path.join(folder_path, filename)

        parts = filename.split('_')
        age = int(parts[0])
        gender = int(parts[1])

        gender_dict = {0:'Male', 1:'Female'}

        test_image = Image.open(image_path).convert('L')  # Open the image in grayscale mode
        test_image = test_image.resize((128, 128), Image.LANCZOS)
        test_image = np.array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0

        # Make predictions
        gender_prob, age_pred = model.predict(test_image)

        # Convert gender prediction probability to actual label
        gender_label = "Male" if gender_prob < 0.5 else "Female"

        # Print the predicted age and gender for the current image
        print("Actual Age:", age)
        print("Actual Gender:", gender_dict[gender])
        print("Predicted Age:", int(round(age_pred[0][0])))
        print("Predicted Gender:", gender_label)
        print()


