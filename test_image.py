from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

# Load the image
img = load_img('test_image/image3.png', target_size=(32, 32))

# Convert the image to a numpy array
img_array = img_to_array(img)

# Reshape the array to match the input shape of the model
img_array = img_array.reshape(1, 32, 32, 3)

# Normalize the pixel values to be between 0 and 1
img_array /= 255.

# Load the model
model = tf.keras.models.load_model('my_model2.h5')

# Make a prediction
pred_probs = model.predict(img_array)

# Print the predicted probabilities for each class
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(len(class_names)):
    print(f'{class_names[i]}: {pred_probs[0][i]}')
