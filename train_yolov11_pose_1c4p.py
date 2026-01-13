#!/usr/bin/env python3
"""
YOLOv11 pose training for the 4-keypoint gauge dataset (yolo_dataset_1c4p).
This script mirrors train_yolov11_pose.py but is configured for the new label
format (bbox + 4 keypoints) produced by the Roboflow dataset.
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

matplotlib.use('Agg')  # non-GUI backend for headless environments

# Patch torch.load BEFORE importing ultralytics to avoid weights_only errors
original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    """Patched torch.load that always sets weights_only=False."""
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)


torch.load = patched_torch_load

try:
    from ultralytics import YOLO
    print("✅ Successfully imported YOLO from ultralytics")
except ImportError as e:
    print(f"❌ Error importing YOLO: {e}")
    print("Make sure you have installed ultralytics:")
    print("pip install ultralytics")
    sys.exit(1)


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "yolo_dataset_1c4p"
DATASET_YAML = DATASET_PATH / "dataset_pose_1c4p.yaml"
NUM_KEYPOINTS = 4
EXPECTED_LABEL_COLUMNS = 5 + NUM_KEYPOINTS * 3  # class + bbox + kpts
CLASS_NAMES = ['gauge']
RUN_PROJECT = Path('runs/pose')
RUN_NAME = 'yolov11_pointer_pose_1c4p'
KEYPOINT_COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 128, 255),
]


class TrainingVisualizer:
    """Utility for saving qualitative results during and after training."""

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.vis_dir = self.save_dir / 'visualizations'
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        print(f"📊 Visualization directory: {self.vis_dir}")

    def save_training_batch_visualization(self, model, dataset_path, epoch, num_samples=4):
        """Run inference on a few validation images and save visualizations."""
        try:
            dataset_path = Path(dataset_path)
            val_images_dir = dataset_path / 'valid' / 'images'

            if not val_images_dir.exists():
                return

            image_files = sorted(val_images_dir.glob('*'))
            image_files = image_files[:num_samples]
            if not image_files:
                return

            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            axes = axes.flatten()

            for idx, img_path in enumerate(image_files):
                if idx >= len(axes):
                    break

                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                annotated = img_rgb.copy()

                results = model.predict(img_path, verbose=False, save=False)
                if not results:
                    axes[idx].imshow(annotated)
                    axes[idx].set_title(f'Epoch {epoch} - {img_path.name}')
                    axes[idx].axis('off')
                    continue

                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f'{conf:.2f}'
                        cv2.putText(
                            annotated,
                            label,
                            (int(x1), max(int(y1) - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                if result.keypoints is not None and len(result.keypoints) > 0:
                    for kpts in result.keypoints:
                        kpts_data = kpts.xy[0].cpu().numpy()
                        prev_point = None
                        for kp_idx, coords in enumerate(kpts_data):
                            if np.any(np.isnan(coords)):
                                continue
                            px, py = int(coords[0]), int(coords[1])
                            color = KEYPOINT_COLORS[kp_idx % len(KEYPOINT_COLORS)]
                            cv2.circle(annotated, (px, py), 9, color, -1)
                            cv2.circle(annotated, (px, py), 15, color, 2)
                            cv2.putText(
                                annotated,
                                str(kp_idx + 1),
                                (px + 4, py - 4),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                color,
                                1,
                            )
                            #if prev_point is not None:
                            #    cv2.line(annotated, prev_point, (px, py), color, 2)
                            prev_point = (px, py)

                axes[idx].imshow(annotated)
                axes[idx].set_title(f'Epoch {epoch} - {img_path.name}')
                axes[idx].axis('off')

            plt.tight_layout()
            if isinstance(epoch, (int, np.integer)):
                epoch_label = f"{int(epoch):03d}"
            else:
                epoch_label = str(epoch)
            save_path = self.vis_dir / f'epoch_{epoch_label}_predictions.jpg'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"  📸 Saved visualization: {save_path.name}")

        except Exception as e:
            print(f"  ⚠️  Visualization error: {e}")

    def plot_metrics(self, metrics_csv: Path):
        """Plot training metrics from results.csv."""
        try:
            if not metrics_csv.exists():
                return

            import pandas as pd

            df = pd.read_csv(metrics_csv)
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            ax = axes[0, 0]
            if 'train/box_loss' in df.columns:
                ax.plot(df['epoch'], df['train/box_loss'], label='Box Loss', linewidth=2)
            if 'train/pose_loss' in df.columns:
                ax.plot(df['epoch'], df['train/pose_loss'], label='Pose Loss', linewidth=2)
            if 'train/kobj_loss' in df.columns:
                ax.plot(df['epoch'], df['train/kobj_loss'], label='Keypoint Obj Loss', linewidth=2)
            if 'train/cls_loss' in df.columns:
                ax.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Losses')
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[0, 1]
            if 'metrics/mAP50(B)' in df.columns:
                ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50 (Box)', linewidth=2, marker='o')
            if 'metrics/mAP50-95(B)' in df.columns:
                ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95 (Box)', linewidth=2, marker='s')
            if 'metrics/mAP50(P)' in df.columns:
                ax.plot(df['epoch'], df['metrics/mAP50(P)'], label='mAP50 (Pose)', linewidth=2, marker='^')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP')
            ax.set_title('Validation mAP')
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[1, 0]
            if 'lr/pg0' in df.columns:
                ax.plot(df['epoch'], df['lr/pg0'], label='LR pg0', linewidth=2)
            if 'lr/pg1' in df.columns:
                ax.plot(df['epoch'], df['lr/pg1'], label='LR pg1', linewidth=2)
            if 'lr/pg2' in df.columns:
                ax.plot(df['epoch'], df['lr/pg2'], label='LR pg2', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.legend()
            ax.grid(True, alpha=0.3)

            ax = axes[1, 1]
            if 'metrics/precision(B)' in df.columns:
                ax.plot(df['epoch'], df['metrics/precision(B)'], label='Precision (Box)', linewidth=2)
            if 'metrics/recall(B)' in df.columns:
                ax.plot(df['epoch'], df['metrics/recall(B)'], label='Recall (Box)', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Score')
            ax.set_title('Precision & Recall')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            save_path = self.vis_dir / 'training_metrics.jpg'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  📈 Saved metrics plot: {save_path.name}")

        except Exception as e:
            print(f"  ⚠️  Metrics plotting error: {e}")

def verify_pose_labels(dataset_path: Path, expected_keypoints: int = NUM_KEYPOINTS) -> bool:
    """Verify that pose label files follow the expected YOLO format."""
    print("\n" + "=" * 60)
    print("Verifying Pose Label Files")
    print("=" * 60)

    dataset_path = Path(dataset_path)
    expected_columns = 5 + expected_keypoints * 3

    for split in ['train', 'valid']:
        labels_dir = dataset_path / split / 'labels'
        if not labels_dir.exists():
            print(f"❌ {split} labels directory not found: {labels_dir}")
            return False

        label_files = sorted(labels_dir.glob('*.txt'))
        if not label_files:
            print(f"❌ No label files found in {labels_dir}")
            return False

        print(f"✅ {split}: Found {len(label_files)} label files")

        for label_file in label_files[:5]:
            with open(label_file, 'r') as f:
                for line_num, line in enumerate(f, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    parts = stripped.split()
                    if len(parts) != expected_columns:
                        print(
                            f"   ⚠️  {label_file.name}:{line_num} has {len(parts)} columns "
                            f"(expected {expected_columns})"
                        )
                        return False
                    try:
                        values = list(map(float, parts))
                    except ValueError:
                        print(f"   ⚠️  Non-numeric values in {label_file.name}:{line_num}")
                        return False

                    bbox = values[1:5]
                    if not all(0.0 <= v <= 1.0 for v in bbox):
                        print(f"   ⚠️  BBox values out of range in {label_file.name}:{line_num}")
                        return False

                    vis_values = values[5 + 2::3]
                    if not all(v in {0.0, 1.0, 2.0} for v in vis_values):
                        print(f"   ⚠️  Visibility values invalid in {label_file.name}:{line_num}")
                        return False
                    break
                else:
                    print(f"   ⚠️  {label_file.name} contains no annotations")
                    return False

        print(f"   ✅ Format verified: {expected_columns} columns ({expected_keypoints} keypoints)")

    return True


def create_pose_dataset_yaml(dataset_path: Path, output_yaml: Path) -> Path:
    """Create a dataset YAML tailored for the 4-keypoint pose dataset."""
    dataset_path = Path(dataset_path)
    output_yaml = Path(output_yaml)

    train_images = len(list((dataset_path / 'train' / 'images').glob('*')))
    valid_images = len(list((dataset_path / 'valid' / 'images').glob('*')))
    test_images = len(list((dataset_path / 'test' / 'images').glob('*')))
    train_labels = len(list((dataset_path / 'train' / 'labels').glob('*.txt')))
    valid_labels = len(list((dataset_path / 'valid' / 'labels').glob('*.txt')))

    config = {
        'path': str(dataset_path.resolve()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES,
        'kpt_shape': [NUM_KEYPOINTS, 3],
        'flip_idx': list(range(NUM_KEYPOINTS)),
    }

    with open(output_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print("\n" + "=" * 60)
    print("Dataset Configuration")
    print("=" * 60)
    print(f"Path: {dataset_path}")
    print(f"Train images: {train_images} (labels: {train_labels})")
    print(f"Valid images: {valid_images} (labels: {valid_labels})")
    print(f"Test images: {test_images}")
    print(f"Classes: {len(CLASS_NAMES)} -> {CLASS_NAMES}")
    print(f"Keypoints per instance: {NUM_KEYPOINTS}")
    print(f"Config saved: {output_yaml}")

    return output_yaml

def train_yolov11_pose() -> bool:
    """Train YOLOv11 pose model on the 4-keypoint dataset."""
    dataset_path = DATASET_PATH
    dataset_yaml = DATASET_YAML

    if not verify_pose_labels(dataset_path, NUM_KEYPOINTS):
        print("\n❌ Pose label verification failed")
        return False

    create_pose_dataset_yaml(dataset_path, dataset_yaml)

    config = {
        'data': str(dataset_yaml),
        'epochs': 100,
        'imgsz': 960,
        'batch': 16,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 2.0,
        'fliplr': 0.0,
        'flipud': 0.0,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'mosaic': 1.0,
        'val': True,
        'plots': True,
        'save': True,
        'save_period': 5,
        'device': '',
        'workers': 8,
        'project': str(RUN_PROJECT),
        'name': RUN_NAME,
        'exist_ok': False,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'amp': True,
        'close_mosaic': 10,
    }

    try:
        print("\n" + "=" * 60)
        print("Initializing YOLOv11-Pose Model")
        print("=" * 60)

        model_variants = ['yolo11s-pose.pt']
        model = None
        loaded_variant = None

        for variant in model_variants:
            try:
                print(f"Trying to load {variant}...")
                model = YOLO(variant)
                loaded_variant = variant
                print(f"✅ Successfully loaded {variant}")
                break
            except Exception as e:
                print(f"⚠️  {variant} not available: {str(e)[:80]}")

        if model is None:
            print("\n📥 Downloading YOLOv11n-pose...")
            try:
                model = YOLO('yolo11n-pose.pt')
                loaded_variant = 'yolo11n-pose.pt'
                print("✅ Downloaded yolo11n-pose.pt")
            except Exception:
                print("📥 Trying YOLOv8n-pose as fallback...")
                model = YOLO('yolov8n-pose.pt')
                loaded_variant = 'yolov8n-pose.pt'
                print("✅ Downloaded yolov8n-pose.pt")

        print("\n" + "=" * 60)
        print("Training Configuration")
        print("=" * 60)
        print(f"Model: {loaded_variant}")
        print(f"Dataset: {dataset_yaml}")
        print(f"Epochs: {config['epochs']}")
        print(f"Batch size: {config['batch']}")
        print(f"Image size: {config['imgsz']}")
        print(f"Keypoints per instance: {NUM_KEYPOINTS}")
        print(f"Pose loss weight: {config['pose']}")
        print(f"Keypoint obj weight: {config['kobj']}")
        print(f"Augmentation: fliplr={config['fliplr']}, flipud={config['flipud']}")
        print("=" * 60)

        save_dir = RUN_PROJECT / RUN_NAME
        visualizer = TrainingVisualizer(save_dir)

        print("\n🚀 Starting training...\n")
        results = model.train(**config)

        print("\n📈 Generating final metrics plots...")
        results_dir = Path(results.save_dir)
        metrics_csv = results_dir / 'results.csv'
        if metrics_csv.exists():
            visualizer.plot_metrics(metrics_csv)

        print("\n🎯 Generating final predictions visualization...")
        best_model = YOLO(results_dir / 'weights' / 'best.pt')
        visualizer.save_training_batch_visualization(best_model, dataset_path, 'final', num_samples=8)

        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        print("=" * 60)
        print(f"📁 Results directory: {results.save_dir}")
        print(f"🏆 Best weights: {results.save_dir}/weights/best.pt")
        print(f"📝 Last weights: {results.save_dir}/weights/last.pt")
        print(f"📊 Visualizations: {visualizer.vis_dir}")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Entry point for launching training on the 1c4p dataset."""
    print("=" * 60)
    print("YOLOv11-Pose Training for Dial Gauge Pointer Detection")
    print("Training: BBox + 4 Keypoints")
    print("With Training Visualization")
    print("=" * 60)

    dataset_path = DATASET_PATH

    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return

    required_dirs = [
        'train/images',
        'train/labels',
        'valid/images',
        'valid/labels',
        'test/images',
        'test/labels',
    ]
    missing_dirs = [d for d in required_dirs if not (dataset_path / d).exists()]

    if missing_dirs:
        print("❌ Error: Required directories not found:")
        for d in missing_dirs:
            print(f"   - {dataset_path / d}")
        return

    print("✅ Dataset structure verified")

    train_images = len(list((dataset_path / 'train/images').glob('*')))
    valid_images = len(list((dataset_path / 'valid/images').glob('*')))
    test_images = len(list((dataset_path / 'test/images').glob('*')))
    train_labels = len(list((dataset_path / 'train/labels').glob('*.txt')))
    valid_labels = len(list((dataset_path / 'valid/labels').glob('*.txt')))

    print(f"\n📊 Dataset Statistics:")
    print(f"  Training: {train_images} images, {train_labels} labels")
    print(f"  Validation: {valid_images} images, {valid_labels} labels")
    print(f"  Test: {test_images} images")

    if train_labels == 0 or valid_labels == 0:
        print("\n❌ Missing labels in train/valid splits")
        return

    label_iter = (dataset_path / 'train/labels').glob('*.txt')
    sample_label = next(label_iter, None)
    if sample_label:
        with open(sample_label, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                parts = first_line.split()
                if len(parts) != EXPECTED_LABEL_COLUMNS:
                    print("\n❌ Label format mismatch!")
                    print(f"  Expected: {EXPECTED_LABEL_COLUMNS} columns ({NUM_KEYPOINTS} keypoints)")
                    print(f"  Actual: {len(parts)} columns")
                    print(f"  Example file: {sample_label.name}")
                    return
                else:
                    print(f"  ✅ Sample label verified ({EXPECTED_LABEL_COLUMNS} columns)")

    success = train_yolov11_pose()

    if success:
        print("\n" + "=" * 60)
        print("🎉 Training completed successfully!")
        print("=" * 60)
        print("\n📂 Output files:")
        print(f"  - {RUN_PROJECT / RUN_NAME / 'weights' / 'best.pt'}")
        print(f"  - {RUN_PROJECT / RUN_NAME / 'weights' / 'last.pt'}")
        print(f"  - {RUN_PROJECT / RUN_NAME / 'visualizations'}/*.jpg")
        print("\n🔍 To use the trained model:")
        print("  from ultralytics import YOLO")
        print(f"  model = YOLO('{RUN_PROJECT / RUN_NAME / 'weights' / 'best.pt'}')")
        print("  results = model.predict('test_image.jpg')")
        print("=" * 60)


if __name__ == "__main__":
    main()
