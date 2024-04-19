from flask import Flask, request, jsonify
from flask_cors import CORS
from inferenceModel import ImageToWordModel 
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

from mltu.configs import BaseModelConfigs
configs = BaseModelConfigs.load("Models/03_handwriting_recognition/202301131202/configs.yaml")
model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform inference on the image
    prediction_text = model.predict(image)

    return jsonify({'prediction': prediction_text})

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)
