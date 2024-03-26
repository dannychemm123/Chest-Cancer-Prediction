from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS
from cnnClassifier.utils.common import decodeImage
from cnnClassifier.pipeline.prediction import PredictionPipeline

# Set environment variables
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self, model_path):
        self.image_filename = 'inputImage.jpg'  # Temporary file name for saving decoded images
        self.classifier = PredictionPipeline(model_path)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def trainRoute():
    os.system('python main.py')
    return 'Training done successfully'

@app.route('/predict', methods=['POST'])
def predictRoute():
    image_data = request.json['image']
    decodeImage(image_data, clApp.image_filename)
    result = clApp.classifier.predict(clApp.image_filename)  # Assuming predict method takes image file path
    return jsonify(result)

if __name__ == '__main__':
    model_path = os.path.join("model", "model.h5")  # Provide the correct path to your trained model
    clApp = ClientApp(model_path)
    app.run(host='0.0.0.0', port=8080)
