import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers, models
import matplotlib.pyplot as plt

def create_model(num_classes):
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(1024, 1024, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Adjust for the number of classes

    return model

# Load dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    'C:/Users/ADMIN/Desktop/Uni/Year 3/AI in Robotics/UnoDetectionProject/dataset/images/train',  # Path to your training data
    image_size=(1024, 1024),  # Resize images to 1024x1024
    batch_size=32,  # Batch size
    label_mode='categorical'  # Use 'categorical' for multi-class classification
)

# Load testing dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    'C:/Users/ADMIN/Desktop/Uni/Year 3/AI in Robotics/UnoDetectionProject/dataset/images/train',  # Path to your testing data
    image_size=(1024, 1024),  # Resize images to 1024x1024
    batch_size=32,  # Batch size
    label_mode='categorical'  # Use 'categorical' for multi-class classification
)

# Optional: visualize the class names
class_names = train_dataset.class_names
print(class_names)

# Number of classes
num_classes = len(class_names)

# Create and compile the model
model = create_model(num_classes)
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


# Train the model
num_epochs = 100  # Set the number of epochs you want to train for

history = model.fit(
    train_dataset,
    epochs=num_epochs
)

print("Training Finished!")


model.save('uno_card_model.h5')

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Visualization of training results
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()