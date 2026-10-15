"""Offline API contract tests with an injected fake predictor (no real ONNX model)."""
import io
import unittest

import cv2
import numpy as np

from app import create_app, MAX_IMAGE_BYTES


class FakePredictor:
    def __init__(self):
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        assert image is not None
        return "sample transcription"


class UploadTests(unittest.TestCase):
    def setUp(self):
        self.model = FakePredictor()
        self.app = create_app(predictor=self.model)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def upload(self, content, filename="example.png", headers=None):
        return self.client.post(
            "/upload",
            data={"file": (io.BytesIO(content), filename)},
            headers=headers or {},
        )

    def test_missing_file_returns_400(self):
        response = self.client.post("/upload")
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.model.calls, 0)

    def test_empty_and_invalid_image_rejected(self):
        self.assertEqual(self.upload(b"").status_code, 400)
        self.assertEqual(self.upload(b"not an image").status_code, 400)
        self.assertEqual(self.upload(b"\x89PNG\r\n\x1a\nBAD").status_code, 400)
        self.assertEqual(self.model.calls, 0)

    def test_real_png_calls_predictor(self):
        pixels = np.full((12, 12, 3), 255, dtype=np.uint8)
        encoded, buffer = cv2.imencode(".png", pixels)
        self.assertTrue(encoded)
        response = self.upload(buffer.tobytes())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"prediction": "sample transcription"})
        self.assertEqual(self.model.calls, 1)

    def test_content_length_limit(self):
        too_big = b"\x89PNG\r\n\x1a\n" + b"0" * (MAX_IMAGE_BYTES + 1)
        self.assertEqual(self.upload(too_big).status_code, 413)
        self.assertEqual(self.model.calls, 0)

    def test_cross_origin_not_public(self):
        response = self.client.post(
            "/upload", headers={"Origin": "https://some-other-site.example"}
        )
        self.assertNotIn("Access-Control-Allow-Origin", response.headers)
        allowed = self.client.post(
            "/upload", headers={"Origin": "http://localhost:5173"}
        )
        self.assertEqual(
            allowed.headers.get("Access-Control-Allow-Origin"),
            "http://localhost:5173",
        )


if __name__ == "__main__":
    unittest.main()
