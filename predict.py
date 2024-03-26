import numpy as np
import os
from tensorflow.keras.models import load_model
from keras.utils import load_img, img_to_array


class PredictionPipeline:
    def __init__(self, model_path, image_size=(224, 224)):
        self.model_path = model_path
        self.image_size = image_size
        self.model = self.load_model()

    def load_model(self):
        """Load and return the model."""
        return load_model(self.model_path)

    def preprocess_image(self, image_path):
        """Preprocess the image to fit the model's input requirements."""
        img = load_img(image_path, target_size=self.image_size)
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, image_path):
        """Predict the class of the image."""
        test_image = self.preprocess_image(image_path)
        result = np.argmax(self.model.predict(test_image), axis=1)
        print(result)

        # Interpret the result
        if result[0] == 1:
            prediction = 'Normal'
        else:
            prediction = 'Adenocarcinoma Cancer'

        return [{"image": prediction}]

# Usage example:
model_path = os.path.join("model", "model.h5")
predictor = PredictionPipeline(model_path)
filename = 'pre.jpg'  # Update this path
prediction = predictor.predict(filename)
print(prediction)
