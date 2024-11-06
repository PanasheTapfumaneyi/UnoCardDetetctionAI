import tensorflow as tf
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from keras import layers, models, Sequential



#Defining the model
def create_model(num_classes):
    #Initialize sequential
    model = models.Sequential()
    #Input layer, the shape is 256,256, with 3 color channels
    model.add(layers.Input(shape=(256, 256, 3)))
    #First convulational layer 64 filters, 3x3 kernel size and relu activation
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    #Max pooling layer to reduce dimentions
    model.add(layers.MaxPooling2D((2, 2)))
    #Second convulational layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    #Dropout layer to reduce overfitting
    model.add(layers.Dropout(0.25))
    #Third Convulational layer
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    #Flatten layer to convert 2D layer to 1D
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(layers.Dropout(0.25))
    #Output layer
    model.add(layers.Dense(num_classes, activation='softmax')) 
    #Return the model
    return model

# Data augmentation layers
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2)
])


# Normalization function to normalize to [0,1]
def normalize_image(image, label):
    image = tf.cast(image, tf.float32)
    return image / 255.0, label  

#load the training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    'C:/Users/ADMIN/Downloads/Test/dataset/images/train',  
    image_size=(256, 256),  # Resize images to 512x512
    batch_size=32,  # Batch size
    label_mode='categorical',  # Categorical for multi-class classification
    color_mode='rgb', #Defining color mode
    validation_split=0.2, #Splitting the dataset for validation
    subset="training",
    seed = 123,
)



# Load testing dataset 
test_dataset = tf.keras.utils.image_dataset_from_directory(
    'C:/Users/ADMIN/Downloads/Test/dataset/images/test', 
    image_size=(256, 256),  
    batch_size=32, 
    label_mode='categorical',  
    color_mode='rgb',   
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    'C:/Users/ADMIN/Downloads/Test/dataset/images/train', 
    image_size=(256, 256),  
    batch_size=32, 
    label_mode='categorical',  
    color_mode='rgb',  
    validation_split=0.2,
    subset="validation",
    seed= 123
)

#Printing the class names
class_names = train_dataset.class_names
print(class_names)


#Calling the normalization function
train_dataset = train_dataset.map(normalize_image)
test_dataset = test_dataset.map(normalize_image)

# Data augmentation calling
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# Defing the number of classes
num_classes = len(class_names)

# Create and compile the model based on the classes
model = create_model(num_classes)
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


# Training the model 

#Defining number of epochs
num_epochs = 20  

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=num_epochs,
)

print("Training Complete!")

#Save the model in .keras format
model.save('uno_card_model_4.keras')

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
