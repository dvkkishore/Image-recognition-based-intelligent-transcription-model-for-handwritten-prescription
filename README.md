# Handwriting OCR with TensorFlow and ONNX

**Documentation:** [Architecture](#project-architecture) · [Local demo](#run-the-local-demo) · [Evaluation](#reproducible-evaluation-entry-point) · [Technical overview](docs/TECHNICAL_OVERVIEW.md)

An experimental **handwritten text recognition** repository, including training code, historic model artifacts, a Flask inference endpoint and a React/Vite upload interface. The repository name refers to medical prescriptions, but the inspected training script reads **IAM handwritten word samples**. The provided materials do not establish prescription-specific training, clinical accuracy, or safe use with medication instructions.

> **Not a medical device or prescribing aid.** Never use the generated text to identify, dispense or dose medication. A recognition error may change a drug name, strength or instruction.

## Project architecture

```text
IAM handwriting data (externally obtained, not supplied here)
    -> train.py (ImageReader, resizing, augmentation, CTC training)
    -> model.py (CNN/residual blocks + bidirectional LSTM + CTC output)
    -> model.keras and exported ONNX checkpoint
    -> inferenceModel.py (ONNX runtime, resize, CTC decoding)
    -> app.py (Flask /upload endpoint)
    -> app/ (React/Vite upload UI)
```

- `model.py` builds a convolutional/residual feature extractor and bidirectional LSTM sequence model.
- `train.py` constructs an IAM-derived dataset, creates a training/validation split and reports CER/WER metrics through the training code. The script is configured for up to 1,000 epochs with early stopping; **this is not evidence that any specific run completed that many epochs**.
- `inferenceModel.py` uses an exported ONNX model and CTC decoding.
- `Word Recognition/` retains an earlier recognition implementation and tutorial-influenced material. See its existing README for the **PyLessons** tutorial source and IAM attribution.
- `Models/` contains historical experiments, checkpoint files and logs, not a single verified final benchmark.

**Historical artifact mismatch:** The API currently selects `Models/03_handwriting_recognition/202301131202`, whose checked-in `val.csv` references `Datasets/IAM_Sentences/` (sentence samples). The current `train.py` instead reads `Datasets/IAM/ascii/words.txt` and `Datasets/IAM/words/`. Consequently, the present training script and the selected historical checkpoint cannot be assumed to describe the same experiment or yield reproducible results. The other saved `202301111911` checkpoint references `Datasets/IAM_Words/` in its validation CSV.

## Getting started

### Prerequisites

The recorded environment includes Python scientific/ML libraries, TensorFlow, `mltu`, ONNX Runtime, Flask and OpenCV. `requirements.txt` is a **historical environment export** containing Windows-specific packages, not a verified cross-platform install recipe. A Python 3.11 environment with compatible versions is a reasonable starting point, but full legacy-model compatibility **has not been verified**.

Use a virtual environment and install compatible dependencies with your platform's supported wheels. Do not deserialize or load model files you have not verified and trust.

### Run the local demo

From the repository root:

```bash
python app.py
```

This starts a local HTTP API at `http://127.0.0.1:5000`. The first valid image upload lazily loads the checked-in experiment referenced by `Models/03_handwriting_recognition/202301131202/configs.yaml`. That YAML references a relative model directory; the API resolves it against this repository instead of the shell's current directory.

In another terminal:

```bash
cd app
npm ci
npm run dev
```

Open the local Vite address and upload a **PNG or JPEG of up to 5 MB**. The API only allows local development UI origins by default. The browser preview uses a temporary object URL and releases it when the selected file changes.

For a different API URL in a local environment, set `VITE_API_URL` when starting Vite. A public deployment would require a separate security/privacy review, cross-origin policy, and consent/data-retention controls.

### Training

`train.py` expects IAM word annotations and image files at `Datasets/IAM/ascii/words.txt` and `Datasets/IAM/words/`. The dataset is **not included** in this repository. Obtain it lawfully from its provider, follow its usage/license terms, then arrange the directory structure before running:

```bash
python train.py
```

Training is compute-intensive and will create a **new experiment directory** under `Models/04_sentence_recognition/`. Historical ONNX exports, cached artifacts and checkpoints are not overwritten by this refinement. Newly generated model vocabularies are sorted for deterministic character indexing. That change applies only to retraining; it does **not** modify the old checkpoint's saved vocabulary.

### Offline tests and frontend build

```bash
python -m unittest discover -s tests -v
cd app && npm ci && npm run build
```

The PR's CI executes Flask upload validation tests with an injected fake predictor, plus a frontend build. It **does not run the real ONNX model, rerun training, validate CER/WER, or check actual medical prescriptions**.

## Reproducible evaluation entry point

When you have lawfully obtained the **exact images referenced by the historical validation CSV** and installed compatible ONNX dependencies, run:

```bash
python evaluate.py \
  --config Models/03_handwriting_recognition/202301131202/configs.yaml \
  --manifest Models/03_handwriting_recognition/202301131202/val.csv \
  --data-root .
```

`--data-root` must contain the `Datasets/IAM_Sentences/` directory referenced by that manifest. `evaluate.py` computes micro-averaged character error rate (CER) and word error rate (WER) using Levenshtein distance, alongside sample/error counts and the model/manifest paths. The evaluation **fails on missing or undecodable images** instead of silently dropping them. It does not download data or save image contents or per-sample predictions.

**No evaluation numbers are claimed by this PR:** the validation images and consistent machine environment were not available for a complete checkpoint evaluation. The offline tests check the metric math and manifest parsing only.

## Evidence and evaluation

**Verified from source:** the architecture, model-artifact locations, data-loading paths, and existence of metric/loss computation.

**Not verified in this repository audit:** reliable held-out CER/WER values, cross-writer generalization, prescription-domain performance, latency/memory on a target device, consent/rights for every retained sample, and any clinical safety claim. Historic log entries should be analyzed and tied to specific dataset splits/checkpoint SHAs before publishing any numeric results.

For a credible portfolio study, run and archive the included evaluator on a documented held-out IAM split, report number of samples and preprocessing, then separately evaluate real prescription handwriting with properly consented data and domain-specific error analysis. Do not claim clinical performance based on IAM metrics.

## Authorship and data

This project incorporates or builds on recognition examples described in the existing **PyLessons handwriting tutorial** under `Word Recognition/README.md`. IAM dataset materials and upstream library/model/code assets may have separate licensing terms. Preserve original creator attribution and independently verify redistribution rights before publishing additional datasets or pretrained weights. The repository contains historic `Test/` images and training artifacts; inspect their provenance before distribution or reuse.

**No patient records should be uploaded** to a public endpoint, shared CI artifacts, issue attachments or a hosted demo without a documented lawful/privacy-compliant workflow.
