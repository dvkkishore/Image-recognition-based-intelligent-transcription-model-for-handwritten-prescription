# Handwriting OCR: Technical overview

## Motivation

Handwritten text has variable word lengths, inconsistent stroke forms and difficult character boundaries. This project explores converting images of handwriting into text using a sequence-recognition model instead of hand-coding character segmentation.

Although the project name mentions medical prescriptions, the training workflow uses **IAM handwriting**. Performance on this dataset does not establish reliable recognition of medicine names or dosages.

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

## Architecture rationale

CTC permits training against variable-length transcripts without manually aligning every character to an image column. CNN layers learn local visual features, while recurrent layers combine context along the output sequence. ONNX enables a separate inference path from the training framework.

The trade-offs include text-domain mismatch, compute demands, imperfect handling of difficult writing, and dependency/version sensitivity of historical model exports.

## Evidence and evaluation status

The repository includes multiple historical model directories, logs and validation manifests. `evaluate.py` can compute micro-averaged CER/WER from predictions **when the exact source images and compatible dependencies are available**. Those images are not fully bundled, and a complete held-out score was not reproduced during the repository review.

**Checkpoint reproducibility issue:** the selected API checkpoint's saved validation manifest points to IAM sentence images, while the current root training script reads IAM word files. Therefore the script cannot be assumed to recreate that checkpoint.

## Local operation

For local demonstration, start `python app.py`, then run `npm ci && npm run dev` inside `app/`. The interface accepts JPG/PNG images up to the documented size cap. Offline API tests and a frontend build have passed CI, but those checks use a fake predictor and do not establish recognition accuracy.

## Data and source provenance

The `Word Recognition/` folder contains material acknowledging a PyLessons handwriting tutorial; IAM data and pretrained weights have their own provenance and usage terms. The source code does not identify individual authorship of each component. Attribution should be checked against original work records and upstream permissions.

**No clinical or medication decisions** should rely on this demonstrator. Prescription-specific performance and privacy safeguards require separate controlled research.

**Primary sources:** [README](../README.md) · [model](../model.py) · [training](../train.py) · [evaluator](../evaluate.py).


## Source layout

| File or directory | Responsibility |
| --- | --- |
| `model.py` | CNN/residual + bidirectional LSTM architecture |
| `train.py` and `configs.py` | IAM data parsing, training and checkpoint configuration |
| `inferenceModel.py` | ONNX prediction and CTC text decoding |
| `evaluate.py` | CER/WER evaluation against a supplied manifest and actual images |
| `app.py` | Local Flask inference endpoint |
| `app/` | React/Vite image-upload demo |
| `Models/` | Historical model artifacts; do not infer that all belong to the same training run |
| `Word Recognition/` | Earlier word-recognition/tutorial-derived code with existing attribution |

## Running or evaluating

The [README](../README.md) describes local server and frontend startup and the required dependencies. Inference requires the selected historical model's artifacts; evaluation also requires the corresponding IAM images to match the saved validation manifest. To run the isolated API contract tests, use `python -m unittest discover -s tests -v`.

Before publishing model performance, confirm that a model checkpoint, source dataset, writer split, vocabulary, metric implementation and manifest refer to the **same experiment**. Current training and the selected historical sentence checkpoint are not guaranteed to match.

## Operational boundaries

The demo was not evaluated for prescription reading, medication interpretation or patient data handling. Do not put identifiable medical records into the demo or treat an OCR guess as a medical decision.
