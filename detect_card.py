import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model('uno_card_model.h5')

# Function to load and preprocess the image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(1024, 1024))  # Resize to the model's input shape
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to predict the class of the card
def predict_card(img_path):
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)  # Get predictions
    predicted_class = np.argmax(predictions, axis=1)  # Get the index of the highest probability
    return predicted_class[0], predictions  # Return the predicted class index and the predictions

# Path to the image you want to test
image_path = 'Red_9.jpg'  # Replace with your image path

# Make a prediction
predicted_class_index, predictions = predict_card(image_path)

# Optional: visualize the input image and predicted class
class_names = [
    'red_0', 'red_1', 'red_2', 'red_3', 'red_4', 'red_5', 'red_6', 'red_7', 'red_8', 'red_9',
    'yellow_0', 'yellow_1', 'yellow_2', 'yellow_3', 'yellow_4', 'yellow_5', 'yellow_6', 'yellow_7', 'yellow_8', 'yellow_9',
    'green_0', 'green_1', 'green_2', 'green_3', 'green_4', 'green_5', 'green_6', 'green_7', 'green_8', 'green_9',
    'blue_0', 'blue_1', 'blue_2', 'blue_3', 'blue_4', 'blue_5', 'blue_6', 'blue_7', 'blue_8', 'blue_9',
    'red_skip', 'red_reverse', 'red_draw_two',
    'yellow_skip', 'yellow_reverse', 'yellow_draw_two',
    'green_skip', 'green_reverse', 'green_draw_two',
    'blue_skip', 'blue_reverse', 'blue_draw_two',
    'wild', 'wild_draw_four'
]  
predicted_class_name = class_names[predicted_class_index]

# Load and display the image
img = image.load_img(image_path, target_size=(1024, 1024))
plt.imshow(img)
plt.title(f'Predicted Class: {predicted_class_name}')
plt.axis('off')
plt.show()

print(f"Predicted Class Index: {predicted_class_index}")
print(f"Prediction Probabilities: {predictions}")
