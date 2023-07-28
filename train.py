# Import necessary libraries
import os
import numpy as np
import pandas as pd
from keras.utils import load_img
from keras.models import Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input
from PIL import Image

# Define data path
path = "/Users/walau/OneDrive/Desktop/model/train_dataset"

# Labels - age, gender, features
age_labels = []
gender_labels = []
features = []

for filename in os.listdir(path):
    # Skip non-image files
    if not filename.endswith(('.jpg', '.png', '.bmp')):
        continue

    image_path = os.path.join(path, filename)
    parts = filename.split('_')
    age = int(parts[0])
    gender = int(parts[1])

    img = load_img(image_path, color_mode="grayscale")
    img = img.resize((128, 128), Image.LANCZOS)
    img = np.array(img)

    features.append(img)
    age_labels.append(age)
    gender_labels.append(gender)
    
# Convert lists to arrays  
features = np.array(features)
y_gender = np.array(gender_labels)
y_age = np.array(age_labels)

# # Data analysis on dataset images
# # Define the age ranges and corresponding labels
# age_ranges = [(10, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 81)]
# labels = ["10 to 19", "20 to 29", "30 to 39" , "40 to 49" , "50 to 59" , "60 to 69" , "70 ++"]

# # Create a new column to store the age range labels
# age_range_labels = pd.cut(y_age, bins=[range_[0] for range_ in age_ranges] + [age_ranges[-1][1]], labels=labels, right=False)

# # Convert y_gender values to "Male" and "Female"
# gender_labels = ["Male" if gender == 0 else "Female" for gender in y_gender]

# # Save the age and gender from image dataset to excel file
# df = pd.DataFrame({"age" : age_range_labels, "gender" : gender_labels})
# df.to_csv("age_and_gender.csv", index=False)

# Reshape features array
features = features.reshape(len(features), 128, 128, 1)

# Normalize the images
features = features / 255.0

# Define input shape
input_shape = (128, 128, 1)

# Define input layer
inputs = Input((input_shape))

# Convolutional layers
conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu') (inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2)) (conv_1)
conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu') (maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2)) (conv_2)
conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu') (maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2)) (conv_3)
conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu') (maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2)) (conv_4)

flatten = Flatten() (maxp_4)

# Fully connected layers
dense_1 = Dense(256, activation='relu') (flatten)
dense_2 = Dense(256, activation='relu') (flatten)

dropout_1 = Dropout(0.3) (dense_1)
dropout_2 = Dropout(0.3) (dense_2)

output_1 = Dense(1, activation='sigmoid', name='gender_out') (dropout_1)
output_2 = Dense(1, activation='relu', name='age_out') (dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy', 'mae'], optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(x=features, y=[y_gender, y_age], batch_size=32, epochs=50, validation_split=0.3)

# Save the trained model
model.save('age_gender_model_50epochs.h5')
