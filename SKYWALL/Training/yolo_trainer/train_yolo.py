#!/usr/bin/env python3
"""
train_yolo.py
SKYWALL - Autonomous Aerial Detection System

YOLOv8-nano training for aerial object detection.
Classes: drone_multirotor, drone_fixed_wing, bird, aircraft, helicopter, balloon, unknown_aerial

Dataset layout (YOLO format):
  datasets/
    aerial/
      images/
        train/   *.jpg / *.png
        val/
        test/
      labels/
        train/   *.txt  (YOLO format: class cx cy w h per line, normalized)
        val/
        test/
      dataset.yaml

Usage:
  python train_yolo.py --data datasets/aerial/dataset.yaml --output ./runs/train
  python train_yolo.py --export_only --weights runs/train/best.pt --output ./export
"""

import argparse
import os
import sys
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Ultralytics YOLOv8
from ultralytics import YOLO
from ultralytics.utils import LOGGER

# CoreML Tools
import coremltools as ct
import torch


# ─── Class Definitions ────────────────────────────────────────────────────────

CLASSES = [
    "drone_multirotor",    # 0
    "drone_fixed_wing",    # 1
    "bird",                # 2
    "aircraft",            # 3
    "helicopter",          # 4
    "balloon",             # 5
    "unknown_aerial",      # 6
]

NUM_CLASSES = len(CLASSES)

# ─── Training Configuration ───────────────────────────────────────────────────

TRAIN_CONFIG: Dict[str, Any] = {
    # Model
    "model": "yolov8n.pt",           # YOLOv8 nano pretrained backbone

    # Training hyperparameters
    "epochs":         200,
    "imgsz":          640,
    "batch":          16,             # Adjust based on GPU memory
    "workers":        8,
    "device":         "auto",         # "auto", "cpu", "cuda:0", "mps"

    # Optimizer
    "optimizer":      "AdamW",
    "lr0":            0.001,
    "lrf":            0.01,           # Final LR = lr0 * lrf
    "momentum":       0.937,
    "weight_decay":   0.0005,
    "warmup_epochs":  5,
    "warmup_momentum": 0.8,

    # Augmentation
    "hsv_h":          0.015,          # Hue augmentation
    "hsv_s":          0.7,
    "hsv_v":          0.4,
    "degrees":        15.0,           # Rotation degrees
    "translate":      0.1,
    "scale":          0.9,            # Scale augmentation
    "shear":          5.0,
    "perspective":    0.0,
    "flipud":         0.0,
    "fliplr":         0.5,
    "mosaic":         1.0,
    "mixup":          0.15,
    "copy_paste":     0.1,

    # Loss weights
    "box":            7.5,
    "cls":            0.5,
    "dfl":            1.5,

    # Other
    "patience":       30,             # Early stopping patience
    "save_period":    10,
    "val":            True,
    "plots":          True,
    "verbose":        True,
    "exist_ok":       True,
}


# ─── Dataset YAML Generator ───────────────────────────────────────────────────

def generate_dataset_yaml(data_root: Path, output_path: Path) -> Path:
    """Generate a YOLO-format dataset.yaml file."""
    config = {
        "path": str(data_root.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    NUM_CLASSES,
        "names": {i: cls for i, cls in enumerate(CLASSES)},
    }

    with open(str(output_path), "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[Dataset] YAML written to {output_path}")
    return output_path


def create_sample_dataset(output_root: Path, n_per_class: int = 20):
    """
    Create a minimal synthetic dataset for pipeline testing.
    In production, replace with real annotated aerial footage.
    Each image: 640x640 sky image, synthetic bounding boxes.
    """
    print(f"[Dataset] Creating synthetic test dataset at {output_root}")

    for split in ["train", "val", "test"]:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    for split, n in [("train", n_per_class * 10), ("val", n_per_class * 2), ("test", n_per_class)]:
        for img_idx in range(n):
            # Create synthetic sky image
            img = Image.fromarray(
                (rng.uniform(140, 200, (640, 640, 3)) * np.array([0.7, 0.85, 1.0])).astype(np.uint8)
            )

            img_path = output_root / "images" / split / f"synthetic_{img_idx:05d}.jpg"
            img.save(str(img_path), quality=85)

            # Synthetic annotation (1-3 objects per image)
            labels = []
            n_objects = rng.integers(1, 4)
            for _ in range(n_objects):
                cls_idx = rng.integers(0, NUM_CLASSES)
                # Small objects (drones are typically 1-5% of frame width)
                cx = rng.uniform(0.1, 0.9)
                cy = rng.uniform(0.05, 0.6)
                w  = rng.uniform(0.015, 0.08)
                h  = rng.uniform(0.015, 0.06)
                labels.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            label_path = output_root / "labels" / split / f"synthetic_{img_idx:05d}.txt"
            with open(str(label_path), "w") as f:
                f.write("\n".join(labels))

    yaml_path = output_root / "dataset.yaml"
    generate_dataset_yaml(output_root, yaml_path)
    print(f"[Dataset] Synthetic dataset ready: {output_root}")
    return yaml_path


# ─── Training ─────────────────────────────────────────────────────────────────

def train(data_yaml: Path, output_dir: Path, config: Dict[str, Any],
          resume_weights: Optional[Path] = None) -> Path:
    """Run YOLOv8 training."""

    model_path = str(resume_weights) if resume_weights else config["model"]
    model = YOLO(model_path)

    print(f"\n[Training] Starting YOLOv8 training")
    print(f"  Model:   {model_path}")
    print(f"  Data:    {data_yaml}")
    print(f"  Output:  {output_dir}")
    print(f"  Classes: {NUM_CLASSES} ({', '.join(CLASSES)})")

    results = model.train(
        data=str(data_yaml),
        project=str(output_dir.parent),
        name=output_dir.name,
        **{k: v for k, v in config.items() if k != "model"},
    )

    best_weights = output_dir / "weights" / "best.pt"
    print(f"\n[Training] Complete. Best weights: {best_weights}")
    return best_weights


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(weights: Path, data_yaml: Path, output_dir: Path):
    """Run model evaluation on test split."""
    print(f"\n[Eval] Evaluating {weights} on test set...")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        project=str(output_dir),
        name="eval",
        conf=0.3,
        iou=0.5,
        plots=True,
        verbose=True,
    )

    print("\n[Eval] Metrics:")
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")

    # Per-class metrics
    print("\n[Eval] Per-class AP50:")
    for i, (cls_name, ap50) in enumerate(zip(CLASSES, metrics.box.ap50)):
        print(f"  {cls_name:25s}: {ap50:.4f}")

    # Save metrics JSON
    metrics_dict = {
        "map50":     float(metrics.box.map50),
        "map50_95":  float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
        "per_class_ap50": {cls: float(ap) for cls, ap in zip(CLASSES, metrics.box.ap50)},
    }
    with open(str(output_dir / "eval_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=2)

    return metrics


# ─── CoreML Export ────────────────────────────────────────────────────────────

def export_to_coreml(weights: Path, output_dir: Path) -> Path:
    """Export YOLOv8 weights to CoreML format for iOS deployment."""
    print(f"\n[Export] Exporting to CoreML...")

    model = YOLO(str(weights))

    # Export to CoreML
    # This generates a .mlpackage in the same directory as weights
    export_path = model.export(
        format="coreml",
        imgsz=640,
        half=False,              # float32 for accuracy
        nms=True,                # Include NMS in the model
        iou=0.45,
        conf=0.25,
        simplify=True,
        device="cpu",
    )

    # Move to output directory
    src = Path(export_path)
    if not src.exists():
        # Ultralytics sometimes changes path format
        src = weights.parent / (weights.stem + ".mlpackage")

    dst = output_dir / "SkyWALLYOLO.mlpackage"
    if src.exists():
        shutil.move(str(src), str(dst))
        print(f"[Export] CoreML model saved to: {dst}")
    else:
        print(f"[Export] Warning: Could not find exported file at {src}")
        # Fallback: manual export with coremltools
        dst = manual_coreml_export(model, output_dir)

    return dst


def manual_coreml_export(model: YOLO, output_dir: Path) -> Path:
    """Manual CoreML conversion using coremltools as fallback."""
    print("[Export] Attempting manual CoreML conversion...")

    # Get the PyTorch model
    pt_model = model.model

    # Trace the model
    dummy_input = torch.zeros(1, 3, 640, 640)
    pt_model.eval()

    try:
        traced = torch.jit.trace(pt_model, dummy_input)
        cml_model = ct.convert(
            traced,
            inputs=[ct.ImageType(
                name="image",
                shape=(1, 3, 640, 640),
                scale=1.0 / 255.0,
                bias=[0, 0, 0],
                color_layout=ct.colorlayout.RGB,
            )],
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS17,
        )

        cml_model.short_description = "SKYWALL Aerial Object Detector (YOLOv8n)"
        cml_model.author = "SKYWALL System"
        cml_model.user_defined_metadata["classes"] = json.dumps(CLASSES)
        cml_model.user_defined_metadata["input_size"] = "640"

        dst = output_dir / "SkyWALLYOLO.mlpackage"
        cml_model.save(str(dst))
        print(f"[Export] Manual CoreML export saved to {dst}")
        return dst

    except Exception as e:
        print(f"[Export] Manual export failed: {e}")
        return output_dir / "EXPORT_FAILED"


# ─── Visualization ────────────────────────────────────────────────────────────

def visualize_predictions(weights: Path, image_dir: Path, output_dir: Path, n: int = 16):
    """Run inference on sample images and save visualizations."""
    model = YOLO(str(weights))
    images = list(image_dir.glob("*.jpg"))[:n] + list(image_dir.glob("*.png"))[:n]

    if not images:
        print("[Viz] No images found for visualization.")
        return

    results = model.predict(
        source=images[:n],
        conf=0.25,
        iou=0.45,
        save=True,
        project=str(output_dir),
        name="predictions",
        verbose=False,
    )

    print(f"[Viz] Saved {len(results)} prediction images to {output_dir / 'predictions'}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SKYWALL YOLOv8 Training")
    parser.add_argument("--data",         type=Path, default=None,
                        help="Path to dataset.yaml (or data root to auto-generate)")
    parser.add_argument("--output",       type=Path, default=Path("./runs/skywall"),
                        help="Output directory")
    parser.add_argument("--epochs",       type=int,  default=TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch",        type=int,  default=TRAIN_CONFIG["batch"])
    parser.add_argument("--imgsz",        type=int,  default=640)
    parser.add_argument("--weights",      type=Path, default=None,
                        help="Resume from or evaluate these weights")
    parser.add_argument("--export_only",  action="store_true",
                        help="Export existing weights to CoreML only")
    parser.add_argument("--eval_only",    action="store_true",
                        help="Evaluate existing weights only")
    parser.add_argument("--create_sample_data", action="store_true",
                        help="Create synthetic dataset for testing")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Create synthetic dataset for pipeline testing
    if args.create_sample_data:
        sample_data_dir = args.output / "synthetic_dataset"
        yaml_path = create_sample_dataset(sample_data_dir)
        print(f"\nSynthetic dataset created at: {sample_data_dir}")
        print(f"To train with it: python train_yolo.py --data {yaml_path}")
        if args.data is None:
            args.data = yaml_path

    # If data is a directory (not yaml), look for/generate yaml
    if args.data and args.data.is_dir():
        yaml_path = args.data / "dataset.yaml"
        if not yaml_path.exists():
            yaml_path = generate_dataset_yaml(args.data, yaml_path)
        args.data = yaml_path

    # Export only mode
    if args.export_only:
        if not args.weights or not args.weights.exists():
            print("[Error] --weights required for --export_only")
            sys.exit(1)
        export_to_coreml(args.weights, args.output)
        return

    # Eval only mode
    if args.eval_only:
        if not args.weights or not args.weights.exists():
            print("[Error] --weights required for --eval_only")
            sys.exit(1)
        if not args.data:
            print("[Error] --data required for --eval_only")
            sys.exit(1)
        evaluate(args.weights, args.data, args.output)
        return

    # Need data for training
    if not args.data:
        print("[Error] --data is required. Use --create_sample_data to generate test data.")
        sys.exit(1)

    # Update config
    config = TRAIN_CONFIG.copy()
    config["epochs"] = args.epochs
    config["batch"]  = args.batch
    config["imgsz"]  = args.imgsz

    # Save config
    with open(str(args.output / "train_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Train ───────────────────────────────────────────────────────────────
    best_weights = train(
        data_yaml=args.data,
        output_dir=args.output / "train",
        config=config,
        resume_weights=args.weights,
    )

    # ── Evaluate ────────────────────────────────────────────────────────────
    if (args.data / "images" / "test").exists() or best_weights.exists():
        evaluate(best_weights, args.data, args.output / "eval")

    # ── Visualize ───────────────────────────────────────────────────────────
    test_images = args.data.parent / "images" / "test"
    if test_images.exists():
        visualize_predictions(best_weights, test_images, args.output / "viz")

    # ── Export ──────────────────────────────────────────────────────────────
    coreml_path = export_to_coreml(best_weights, args.output)

    print(f"\n{'='*60}")
    print(f"[Done] Training complete.")
    print(f"  Best weights:  {best_weights}")
    print(f"  CoreML model:  {coreml_path}")
    print(f"  Output dir:    {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
