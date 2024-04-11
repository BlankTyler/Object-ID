import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#Load the pre-trained model
model = tf.keras.models.load_model('image_id.h5')

#Create CIFAR-10 classes
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#Function to preprocess the input image files
def preprocess_image(image_path):
    img = Image.open(image_path)
   
    #Resize the image to the model's input size
    img = img.resize((32, 32))  
    img_array = np.array(img)
   
    #Normalize pixel values to [0, 1]
    img_array = img_array / 255.0  
    return img_array

#Function to make predictions on the input image files
def predict_image(model, image_path):
    img_array = preprocess_image(image_path)
   
    #Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)  
    prediction = model.predict(img_batch)
    predicted_class_index = np.argmax(prediction[0])
    predicted_class = classes[predicted_class_index]
    return predicted_class

#Replace image_file_path with the actual path of the image you want to classify
image_file_path = 'unlabeled_image_png_46.png'
predicted_subject = predict_image(model, image_file_path)

#Display the image with the predicted subject
img = Image.open(image_file_path)
plt.imshow(img)
plt.title(f"Predicted subject: {predicted_subject}")
plt.axis('off')
plt.show()