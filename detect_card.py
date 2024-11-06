import cv2
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = tf.keras.models.load_model('uno_card_model.keras')

# Function to preprocess the image
def load_and_preprocess_image(img):
    img = cv2.resize(img, (512, 512))  # Resize to the same input shape as the model
    img_array = image.img_to_array(img)  # Convert image to array 
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to predict the class of the uno card
def predict_card(frame):
    img_array = load_and_preprocess_image(frame)
    predictions = model.predict(img_array)  # Fetch predictions
    predicted_class = np.argmax(predictions, axis=1)  # Get the index with the highest probability
    return predicted_class[0], predictions  # Return the predictions and the predicted class index

# Prediction Class Names
class_names = ['blue_0', 'blue_1', 'blue_2', 'blue_3', 'blue_4', 'blue_5', 'blue_6', 'blue_7', 'blue_8', 'blue_9', 'blue_draw_two', 
               'blue_reverse', 'blue_skip', 'green_0', 'green_1', 'green_2', 'green_3', 'green_4', 'green_5', 'green_6', 
               'green_7', 'green_8', 'green_9', 'green_draw_two', 'green_reverse', 'green_skip', 'red_0', 'red_1', 'red_2', 
               'red_3', 'red_4', 'red_5', 'red_6', 'red_7', 'red_8', 'red_9', 'red_draw_two', 'red_reverse', 'red_skip', 
               'wild', 'wild_draw_four', 'yellow_0', 'yellow_1', 'yellow_2', 'yellow_3', 'yellow_4', 'yellow_5', 'yellow_6', 
               'yellow_7', 'yellow_8', 'yellow_9', 'yellow_draw_two', 'yellow_reverse', 'yellow_skip']

# Start video capture

cap = cv2.VideoCapture("http://192.168.100.166:8080/video")

while True:
    ret, frame = cap.read() #Capure each frame
    if not ret:
        break

    predicted_class_index, predictions = predict_card(frame)
    predicted_class_name = class_names[predicted_class_index]

    # Display the prediction in the frams
    cv2.putText(frame, f'Predicted Class: {predicted_class_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Card Detection', frame)

    #Press q to quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Close the Windows
cap.release()
cv2.destroyAllWindows()
