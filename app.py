"""Local Flask API for experimental IAM handwriting transcription.

Not validated for interpreting medication or clinical prescriptions.
"""
import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "Models/03_handwriting_recognition/202301131202/configs.yaml"
MAX_IMAGE_BYTES = 5 * 1024 * 1024


def load_predictor():
    """Load the shipped model lazily and resolve artifact paths from the repo."""
    from mltu.configs import BaseModelConfigs
    from inferenceModel import ImageToWordModel

    configs = BaseModelConfigs.load(str(DEFAULT_CONFIG))
    model_path = Path(str(configs.model_path).replace("\\", "/"))
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    return ImageToWordModel(model_path=str(model_path), char_list=configs.vocab)


def create_app(predictor=None):
    api = Flask(__name__)
    api.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES
    CORS(
        api,
        resources={
            r"/upload": {
                "origins": ["http://localhost:5173", "http://127.0.0.1:5173"]
            }
        },
    )

    current_predictor = predictor

    @api.route("/upload", methods=["POST"])
    def upload_file():
        nonlocal current_predictor
        uploaded = request.files.get("file")
        if uploaded is None:
            return jsonify(error="No file uploaded"), 400
        if not uploaded.filename:
            return jsonify(error="No file selected"), 400

        content = uploaded.read()
        if not content:
            return jsonify(error="Image file is empty"), 400

        # Accept only JPEG/PNG image contents, regardless of the filename.
        is_png = content.startswith(b"\x89PNG\r\n\x1a\n")
        is_jpeg = content.startswith(b"\xff\xd8\xff")
        if not (is_png or is_jpeg):
            return jsonify(error="Only JPEG and PNG images are supported"), 400

        import cv2
        import numpy as np

        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify(error="The image could not be decoded"), 400

        if current_predictor is None:
            current_predictor = load_predictor()

        return jsonify(prediction=current_predictor.predict(image))

    return api


app = create_app()


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
