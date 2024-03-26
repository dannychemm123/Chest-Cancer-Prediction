# prediction.py
import tensorflow as tf
import keras.utils as image
import numpy as np
from pathlib import Path

class PredictionPipeline:
    def __init__(self, model_path, image_size=(224, 224)):
        self.model_path = model_path
        self.image_size = image_size
        self.model = self.load_model()

    def load_model(self):
        return tf.keras.models.load_model(self.model_path)

    def preprocess_image(self, image_path):
        img = image.load_img(image_path, target_size=self.image_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add a new axis for batch dimension
        img_array /= 255.  # Assuming model expects pixel values to be scaled between 0 and 1
        return img_array

    def predict(self, image_path):
        processed_image = self.preprocess_image(image_path)
        prediction = self.model.predict(processed_image)
        predicted_class = np.argmax(prediction, axis=1)
        print(predicted_class)

        
        if predicted_class[0] == 1:
            prediction = 'Normal'
        else:
            prediction = 'Adenocarcinoma Cancer'

        return {"image": prediction}


