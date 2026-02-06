import tensorflow as tf
import numpy as np
from PIL import Image
import sys

# Change 'model.h5' if your file name is different!
model = tf.keras.models.load_model('speech_emotion_model.h5')

def run_prediction(image_path):
    # This prepares your image for the model
    img = Image.open(image_path).resize((224, 224)) 
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # This makes the prediction
    prediction = model.predict(img_array)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    # This takes the image name you type in the command line
    run_prediction(sys.argv[1])
