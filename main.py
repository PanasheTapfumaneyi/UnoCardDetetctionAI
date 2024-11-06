import tensorflow as tf
import tensorflow_datasets as tfds
from keras import layers, models
import matplotlib.pyplot as plt 


#Defining the model
def create_model(num_classes):
    model = models.Sequential()
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax')) 

    return model

# Normalization function to normalize to [0,1]
def normalize_image(image, label):
    return image / 255.0, label  

#load the training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    'C:/Users/ADMIN/Downloads/Test/dataset/images/train',  
    image_size=(512, 512),  # Resize images to 512x512
    batch_size=32,  # Batch size
    label_mode='categorical',  # Categorical for multi-class classification
    color_mode='rgb', #Defining color mode
    validation_split=0.2,
    subset="training",
)

# Load testing dataset 
test_dataset = tf.keras.utils.image_dataset_from_directory(
    'C:/Users/ADMIN/Downloads/Test/dataset/images/train', 
    image_size=(512, 512),  
    batch_size=32, 
    label_mode='categorical',  
    color_mode='rgb',   
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    'C:/Users/ADMIN/Downloads/Test/dataset/images/train', 
    image_size=(512, 512),  
    batch_size=32, 
    label_mode='categorical',  
    color_mode='rgb',  
    validation_split=0.2,
    subset="validation",
)

#Printing the class names
class_names = train_dataset.class_names
print(class_names)

#Calling the normalization function
train_dataset = train_dataset.map(normalize_image)
test_dataset = test_dataset.map(normalize_image)

# Defing the number of classes
num_classes = len(class_names)

# Create and compile the model based on the classes
model = create_model(num_classes)
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


# Training the model 

#Defining number of epochs
num_epochs = 10  

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=num_epochs
)

print("Training Complete!")

#Save the model in .keras format
model.save('uno_card_model.keras')

# Evaluating the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}")

# Visualizating the training results
plt.figure(figsize=(12, 4))

# Plotting the model accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plotting the loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
