# OCR demo frontend

A React/Vite proof-of-concept upload form for the repository's experimental IAM handwriting recognizer.

## Local development

Start the Flask API from the repository root with `python app.py`, then run:

```bash
npm ci
npm run dev
```

The form sends a PNG/JPEG image (up to 5 MB) to `http://localhost:5000/upload`. Override with a local `VITE_API_URL` environment variable if required. The API allows Vite origins on `localhost:5173` and `127.0.0.1:5173` by default.

Run `npm run build` to compile the static frontend. A frontend build does not exercise the Python model.

**Safety:** the model is not validated on medical prescriptions. Never use its output to decide medication name, dosage or administration. Do not upload identifiable patient records to an unreviewed service.
