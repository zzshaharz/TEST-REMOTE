#!/usr/bin/env python3
"""
train_sound_classifier.py
SKYWALL - Autonomous Aerial Detection System

Sound classifier training pipeline:
  1. Load audio files from dataset directory
  2. Extract mel spectrogram features with librosa
  3. Data augmentation (pitch shift, time stretch, noise, room IR)
  4. Train a PyTorch CNN classifier
  5. Export to CoreML .mlpackage for on-device inference
  6. Evaluate with confusion matrix and per-class metrics

Dataset layout expected:
  data/
    fpv_5inch/       *.wav / *.mp3
    fpv_7inch/
    fpv_10inch/
    dji_mavic/
    dji_phantom/
    dji_inspire/
    fixed_wing_small/
    fixed_wing_large/
    helicopter/
    bird/
    aircraft/
    wind/
    traffic/
    background/

Usage:
  python train_sound_classifier.py --data_dir ./data --output ./output
"""

import argparse
import os
import sys
import json
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import librosa
import librosa.display
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import coremltools as ct
from tqdm import tqdm

# ─── Constants ────────────────────────────────────────────────────────────────

CLASSES = [
    "fpv_5inch", "fpv_7inch", "fpv_10inch",
    "dji_mavic", "dji_phantom", "dji_inspire",
    "fixed_wing_small", "fixed_wing_large",
    "helicopter", "bird", "aircraft",
    "wind", "traffic", "background",
]

SAMPLE_RATE    = 48_000
WINDOW_SEC     = 0.5      # 500ms
HOP_SEC        = 0.25     # 50% overlap
N_FFT          = 2048
HOP_LENGTH     = 512
N_MELS         = 128
F_MIN          = 50.0
F_MAX          = 12_000.0
NUM_FRAMES     = int(WINDOW_SEC * SAMPLE_RATE / HOP_LENGTH) + 1   # ~47 frames

BATCH_SIZE     = 64
EPOCHS         = 80
LR             = 3e-4
WEIGHT_DECAY   = 1e-4
DROPOUT        = 0.4
TRAIN_SPLIT    = 0.80
VAL_SPLIT      = 0.10
# TEST_SPLIT = 0.10 (remainder)

# ─── Data Augmentation ────────────────────────────────────────────────────────

def augment_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Apply random augmentation pipeline to a waveform."""
    augmentations = []

    # Pitch shift ±2 semitones
    if random.random() < 0.4:
        n_steps = random.uniform(-2.0, 2.0)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        augmentations.append(f"pitch_shift({n_steps:.2f})")

    # Time stretch ×0.85–1.15
    if random.random() < 0.4:
        rate = random.uniform(0.85, 1.15)
        y = librosa.effects.time_stretch(y, rate=rate)
        augmentations.append(f"time_stretch({rate:.2f})")

    # Additive white noise
    if random.random() < 0.5:
        snr_db = random.uniform(15.0, 40.0)
        rms = np.sqrt(np.mean(y ** 2)) + 1e-10
        noise_rms = rms / (10 ** (snr_db / 20.0))
        y = y + np.random.normal(0, noise_rms, len(y)).astype(np.float32)
        augmentations.append(f"noise(snr={snr_db:.0f}dB)")

    # Volume gain ±6 dB
    if random.random() < 0.5:
        gain_db = random.uniform(-6.0, 6.0)
        y = y * (10 ** (gain_db / 20.0))
        augmentations.append(f"gain({gain_db:.1f}dB)")

    # High-pass filter (removes very low freq rumble below 40 Hz)
    if random.random() < 0.3:
        from scipy.signal import butter, filtfilt
        b, a = butter(2, 40.0 / (sr / 2), btype='high')
        y = filtfilt(b, a, y).astype(np.float32)

    # Simulate room reverb via simple convolution reverb
    if random.random() < 0.2:
        reverb_len = int(sr * random.uniform(0.05, 0.3))
        ir = np.random.exponential(0.5, reverb_len).astype(np.float32)
        ir /= np.sum(ir) + 1e-10
        y = np.convolve(y, ir, mode='same').astype(np.float32)
        augmentations.append("reverb")

    return np.clip(y, -1.0, 1.0)


def extract_mel_spectrogram(y: np.ndarray, sr: int, augment: bool = False) -> np.ndarray:
    """Extract log mel spectrogram. Returns shape [n_mels, n_frames]."""
    if augment:
        y = augment_audio(y, sr)

    # Ensure correct length (pad or trim to WINDOW_SEC)
    target_len = int(WINDOW_SEC * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
    else:
        # Random crop during training
        if augment and len(y) > target_len:
            start = random.randint(0, len(y) - target_len)
            y = y[start:start + target_len]
        else:
            y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX,
        power=2.0
    )

    # Log compression
    log_mel = librosa.power_to_db(mel, ref=np.max, top_db=80.0)

    # Normalize to [0, 1]
    log_mel = (log_mel + 80.0) / 80.0

    # Ensure consistent shape
    if log_mel.shape[1] < NUM_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, NUM_FRAMES - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :NUM_FRAMES]

    return log_mel.astype(np.float32)  # [n_mels, n_frames]


# ─── Dataset ──────────────────────────────────────────────────────────────────

class DroneAudioDataset(Dataset):
    def __init__(self, file_list: List[Tuple[Path, int]], augment: bool = False):
        self.file_list = file_list
        self.augment   = augment

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.file_list[idx]
        try:
            y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=WINDOW_SEC * 3)
        except Exception as e:
            print(f"[Warning] Failed to load {path}: {e}. Using silence.")
            y = np.zeros(int(SAMPLE_RATE * WINDOW_SEC), dtype=np.float32)
            sr = SAMPLE_RATE

        mel = extract_mel_spectrogram(y, sr, augment=self.augment)
        # Add channel dimension: [1, n_mels, n_frames]
        return torch.from_numpy(mel).unsqueeze(0), label


def build_file_list(data_dir: Path) -> List[Tuple[Path, int]]:
    """Scan data directory and build (file, label_idx) list."""
    file_list = []
    extensions = {'.wav', '.mp3', '.ogg', '.flac', '.aac', '.m4a'}

    for cls_idx, cls_name in enumerate(CLASSES):
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            print(f"[Warning] Class directory not found: {cls_dir}. Creating empty.")
            cls_dir.mkdir(parents=True, exist_ok=True)
            # Generate synthetic samples for missing classes
            file_list.extend(generate_synthetic_samples(cls_idx, n=50))
            continue

        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in extensions]
        if len(files) == 0:
            print(f"[Warning] No audio files in {cls_dir}. Generating synthetic samples.")
            file_list.extend(generate_synthetic_samples(cls_idx, n=50))
        else:
            for f in files:
                file_list.append((f, cls_idx))
            print(f"  {cls_name:25s}: {len(files):4d} files")

    return file_list


def generate_synthetic_samples(label_idx: int, n: int = 50) -> List[Tuple[Path, int]]:
    """Generate synthetic audio files for missing classes (for testing pipeline)."""
    # Returns dummy entries; in production, real audio data is required
    # We use a sentinel path that the dataset will handle as silence
    return [(Path("/dev/null"), label_idx)] * n


# ─── Model Architecture ───────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.drop  = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class DroneAudioCNN(nn.Module):
    """
    Lightweight CNN for drone audio classification.
    Input: [batch, 1, 128, 47]  (channels, mel_bins, frames)
    Output: [batch, num_classes]
    ~1.2M parameters - suitable for CoreML Neural Engine.
    """
    def __init__(self, num_classes: int = len(CLASSES), dropout: float = DROPOUT):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),       # → [32, 64, 23]
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, dropout=0.2),
            nn.MaxPool2d(kernel_size=(2, 2)),       # → [64, 32, 11]
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128, dropout=0.3),
            nn.MaxPool2d(kernel_size=(2, 2)),       # → [128, 16, 5]
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResidualBlock(256, dropout=0.3),
            nn.AdaptiveAvgPool2d((4, 2)),            # → [256, 4, 2]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(loader, desc="  Train", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += inputs.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> Tuple[float, float, List, List]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(loader, desc="  Eval", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss    = criterion(outputs, labels)
        preds   = outputs.argmax(dim=1)

        total_loss += loss.item() * inputs.size(0)
        correct    += (preds == labels).sum().item()
        total      += inputs.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


def plot_confusion_matrix(cm: np.ndarray, output_path: Path):
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASSES, yticklabels=CLASSES,
        linewidths=0.5
    )
    plt.title("Confusion Matrix - SKYWALL Sound Classifier", fontsize=14)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"[Eval] Confusion matrix saved to {output_path}")


# ─── CoreML Export ────────────────────────────────────────────────────────────

def export_to_coreml(model: nn.Module, output_dir: Path) -> Path:
    model.eval()
    model.cpu()

    # Create example input
    example_input = torch.zeros(1, 1, N_MELS, NUM_FRAMES)

    # Trace model
    traced = torch.jit.trace(model, example_input)

    # Convert to CoreML
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(
            name="melSpectrogram",
            shape=(1, 1, N_MELS, NUM_FRAMES),
            dtype=np.float32
        )],
        outputs=[ct.TensorType(name="classProbs")],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS17,
    )

    # Add metadata
    mlmodel.short_description = "SKYWALL Drone Sound Classifier"
    mlmodel.author = "SKYWALL System"
    mlmodel.version = "1.0"

    # Add class labels to output
    spec = mlmodel.get_spec()

    # Add user-defined metadata
    mlmodel.user_defined_metadata["classes"] = json.dumps(CLASSES)
    mlmodel.user_defined_metadata["sample_rate"] = str(SAMPLE_RATE)
    mlmodel.user_defined_metadata["n_mels"] = str(N_MELS)
    mlmodel.user_defined_metadata["num_frames"] = str(NUM_FRAMES)
    mlmodel.user_defined_metadata["f_min"] = str(F_MIN)
    mlmodel.user_defined_metadata["f_max"] = str(F_MAX)

    mlmodel_path = output_dir / "SkyWALLSoundClassifier.mlpackage"
    mlmodel.save(str(mlmodel_path))
    print(f"[Export] CoreML model saved to {mlmodel_path}")

    # Also save as .mlmodel for older toolchains
    mlmodel_v4_path = output_dir / "SkyWALLSoundClassifier.mlmodel"
    ct.convert(
        traced,
        inputs=[ct.TensorType(name="melSpectrogram", shape=(1, 1, N_MELS, NUM_FRAMES))],
        outputs=[ct.TensorType(name="classProbs")],
    ).save(str(mlmodel_v4_path))

    return mlmodel_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SKYWALL Sound Classifier Training")
    parser.add_argument("--data_dir", type=Path, default=Path("./data"),
                        help="Root directory with class subdirectories")
    parser.add_argument("--output",   type=Path, default=Path("./output"),
                        help="Output directory for model and logs")
    parser.add_argument("--epochs",   type=int,  default=EPOCHS)
    parser.add_argument("--batch",    type=int,  default=BATCH_SIZE)
    parser.add_argument("--lr",       type=float, default=LR)
    parser.add_argument("--resume",   type=Path, default=None,
                        help="Path to checkpoint to resume training")
    parser.add_argument("--eval_only", action="store_true",
                        help="Run evaluation only (requires --resume)")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"[Training] Device: {device}")
    print(f"[Training] Classes: {len(CLASSES)}")

    # ── Build dataset ───────────────────────────────────────────────────────
    print(f"\n[Data] Scanning {args.data_dir}...")
    all_files = build_file_list(args.data_dir)
    random.shuffle(all_files)
    print(f"[Data] Total samples: {len(all_files)}")

    n_total  = len(all_files)
    n_train  = int(n_total * TRAIN_SPLIT)
    n_val    = int(n_total * VAL_SPLIT)
    n_test   = n_total - n_train - n_val

    train_files = all_files[:n_train]
    val_files   = all_files[n_train:n_train + n_val]
    test_files  = all_files[n_train + n_val:]

    print(f"[Data] Split → train={n_train} val={n_val} test={n_test}")

    train_ds = DroneAudioDataset(train_files, augment=True)
    val_ds   = DroneAudioDataset(val_files,   augment=False)
    test_ds  = DroneAudioDataset(test_files,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=2)

    # ── Model ───────────────────────────────────────────────────────────────
    model = DroneAudioCNN(num_classes=len(CLASSES)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Parameters: {n_params:,}")

    # Class weights for imbalanced data
    class_counts = [sum(1 for _, l in all_files if l == i) for i in range(len(CLASSES))]
    class_weights = torch.tensor([1.0 / (c + 1) for c in class_counts], dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(CLASSES)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0

    if args.resume and args.resume.exists():
        ckpt = torch.load(str(args.resume), map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"[Resume] Loaded checkpoint: epoch={start_epoch} best_val_acc={best_val_acc:.4f}")

    if args.eval_only:
        print("\n[Eval] Running test evaluation...")
        _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)
        print(f"[Eval] Test accuracy: {test_acc:.4f}")
        report = classification_report(test_labels, test_preds, target_names=CLASSES, digits=3)
        print(report)
        cm = confusion_matrix(test_labels, test_preds)
        plot_confusion_matrix(cm, args.output / "confusion_matrix.png")
        return

    # ── Training loop ───────────────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc, _, _ = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - epoch_start
        print(f"  loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"lr={scheduler.get_last_lr()[0]:.6f} [{elapsed:.1f}s]")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = args.output / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "classes": CLASSES,
            }, str(ckpt_path))
            print(f"  ✓ Best model saved (val_acc={best_val_acc:.4f})")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "classes": CLASSES,
            }, str(args.output / f"checkpoint_ep{epoch+1:04d}.pth"))

    # ── Final evaluation ────────────────────────────────────────────────────
    print("\n[Eval] Loading best model for final evaluation...")
    best_ckpt = torch.load(str(args.output / "best_model.pth"), map_location=device)
    model.load_state_dict(best_ckpt["model"])

    _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)
    print(f"\n[Eval] Final Test Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

    report = classification_report(test_labels, test_preds, target_names=CLASSES, digits=3)
    print(report)
    with open(str(args.output / "classification_report.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(cm, args.output / "confusion_matrix.png")

    # ── Export to CoreML ────────────────────────────────────────────────────
    print("\n[Export] Converting to CoreML...")
    mlmodel_path = export_to_coreml(model, args.output)
    print(f"[Export] Done: {mlmodel_path}")

    # Save training history
    with open(str(args.output / "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Done] Best val accuracy: {best_val_acc:.4f}")
    print(f"[Done] Output directory: {args.output}")


if __name__ == "__main__":
    main()
