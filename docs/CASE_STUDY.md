# Handwritten text recognition: CNN + BiLSTM + CTC

## Motivation

Handwritten text has variable word lengths, inconsistent stroke forms and difficult character boundaries. This project explores converting images of handwriting into text using a sequence-recognition model instead of hand-coding character segmentation.

Although the repository's long-standing name mentions medical prescriptions, the source-verified training workflow uses **IAM handwriting**. That distinction matters: an IAM recognizer is not automatically trustworthy for medicine names or dosages.

## System architecture

```text
IAM word/sentence image -> resize / augment -> residual CNN features
                                              |
                                        BiLSTM sequence
                                              |
                                       CTC loss + decoder
                                              |
                                  saved model / ONNX inference
                                              |
                            Flask upload endpoint -> React preview
```

- `model.py`: residual convolution blocks, reshaping to time steps, two bidirectional LSTM stages, softmax sequence output.
- `train.py`: IAM-derived annotation parsing, image preprocessing, training/validation split, and CER/WER metric configuration.
- `inferenceModel.py`: ONNX execution and CTC decoding.
- `app.py` + `app/`: local image-upload/inference demonstration.

## Why this approach

CTC permits training against variable-length transcripts without manually aligning every character to an image column. CNN layers learn local visual features, while recurrent layers combine context along the output sequence. ONNX enables a separate inference path from the training framework.

The trade-offs include text-domain mismatch, compute demands, imperfect handling of difficult writing, and dependency/version sensitivity of historical model exports.

## Evidence and evaluation status

The repository includes multiple historical model directories, logs and validation manifests. `evaluate.py` can compute micro-averaged CER/WER from predictions **when the exact source images and compatible dependencies are available**. Those images are not fully bundled, and a complete held-out score was not reproduced during the repository review.

**Checkpoint reproducibility issue:** the selected API checkpoint's saved validation manifest points to IAM sentence images, while the current root training script reads IAM word files. Therefore the script cannot be assumed to recreate that checkpoint.

## Demonstration

For local demonstration, start `python app.py`, then run `npm ci && npm run dev` inside `app/`. The interface accepts JPG/PNG images up to the documented size cap. Offline API tests and a frontend build have passed CI, but those checks use a fake predictor and do not establish recognition accuracy.

## Attribution and responsible framing

The `Word Recognition/` folder contains material acknowledging a PyLessons handwriting tutorial; IAM data and pretrained weights have their own provenance and usage terms. Individual development contributions are not enumerated here, and should be stated accurately before using this as a personal portfolio case study.

**No clinical or medication decisions** should rely on this demonstrator. Prescription-specific performance and privacy safeguards require separate controlled research.

**Primary sources:** [README](../README.md) · [model](../model.py) · [training](../train.py) · [evaluator](../evaluate.py).
