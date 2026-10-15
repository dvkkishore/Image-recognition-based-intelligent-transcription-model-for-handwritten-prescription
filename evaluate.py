"""Evaluate an IAM handwriting checkpoint on an existing annotated image manifest.

No images or per-sample transcriptions are written to disk. This script never
downloads or fabricates data. It fails if referenced image files are missing.
"""
import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "Models/03_handwriting_recognition/202301131202/configs.yaml"
DEFAULT_MANIFEST = ROOT / "Models/03_handwriting_recognition/202301131202/val.csv"


def edit_distance(reference, prediction):
    """Levenshtein distance for sequences of characters or words."""
    previous = list(range(len(prediction) + 1))
    for i, left in enumerate(reference, start=1):
        current = [i]
        for j, right in enumerate(prediction, start=1):
            current.append(
                min(previous[j] + 1, current[-1] + 1, previous[j - 1] + (left != right))
            )
        previous = current
    return previous[-1]


def error_rates(reference_prediction_pairs):
    """Micro-averaged character and word error rates across a labeled corpus."""
    character_errors = character_count = word_errors = word_count = num_samples = 0
    for reference, prediction in reference_prediction_pairs:
        if not isinstance(reference, str) or not isinstance(prediction, str):
            raise TypeError("Reference and prediction must be strings")
        if not reference:
            raise ValueError("Empty ground-truth strings require explicit evaluation policy")
        character_errors += edit_distance(reference, prediction)
        character_count += len(reference)
        words = reference.split()
        if not words:
            raise ValueError("Ground-truth labels must contain words")
        word_errors += edit_distance(words, prediction.split())
        word_count += len(words)
        num_samples += 1
    if not num_samples:
        raise ValueError("No labeled samples to evaluate")
    return {
        "samples": num_samples,
        "cer": character_errors / character_count,
        "wer": word_errors / word_count,
        "character_errors": character_errors,
        "reference_characters": character_count,
        "word_errors": word_errors,
        "reference_words": word_count,
    }


def read_manifest(path):
    """Support the old DataProvider CSV schema ('0','1') and named columns."""
    with open(path, newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        names = set(rows.fieldnames or [])
        if {"image_path", "label"} <= names:
            image_col, label_col = "image_path", "label"
        elif {"0", "1"} <= names:
            image_col, label_col = "0", "1"
        else:
            raise ValueError("Expected columns image_path,label or 0,1")
        for row in rows:
            image_path = (row.get(image_col) or "").strip()
            ground_truth = row.get(label_col)
            if not image_path or ground_truth is None:
                raise ValueError("Missing path or label in the validation manifest")
            yield image_path, ground_truth


def evaluate(manifest, config, data_root):
    import cv2
    from mltu.configs import BaseModelConfigs
    from inferenceModel import ImageToWordModel

    configs = BaseModelConfigs.load(str(config))
    model_dir = Path(str(configs.model_path).replace("\\", "/"))
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    predictor = ImageToWordModel(model_path=str(model_dir), char_list=configs.vocab)
    pairs = []
    for relative_name, label in read_manifest(manifest):
        image_path = Path(relative_name.replace("\\", "/"))
        if not image_path.is_absolute():
            image_path = data_root / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Required validation image not found: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Image could not be decoded: {image_path}")
        pairs.append((label, predictor.predict(image)))
    result = error_rates(pairs)
    result["checkpoint_config"] = str(config)
    result["validation_manifest"] = str(manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--data-root", type=Path, default=ROOT,
        help="Directory containing the Datasets/ folder referenced by the CSV",
    )
    args = parser.parse_args()
    print(json.dumps(evaluate(args.manifest, args.config, args.data_root), indent=2))


if __name__ == "__main__":
    main()
