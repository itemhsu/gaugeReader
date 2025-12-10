#!/usr/bin/env python3
"""
YOLOv11-Pose Training Script for Dial Gauge Pointer Detection
Trains on bbox + 2 keypoints with visualization
"""
import os
import sys
import torch
import yaml
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非GUI後端

# Patch torch.load BEFORE importing ultralytics
original_torch_load = torch.load

def patched_torch_load(*args, **kwargs):
    """Patched torch.load that sets weights_only=False"""
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)

torch.load = patched_torch_load

try:
    from ultralytics import YOLO
    from ultralytics.utils.plotting import Annotator, colors
    print("✅ Successfully imported YOLO from ultralytics")
except ImportError as e:
    print(f"❌ Error importing YOLO: {e}")
    print("Make sure you have installed ultralytics:")
    print("pip install ultralytics")
    sys.exit(1)


class TrainingVisualizer:
    """訓練過程可視化器"""
    
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.vis_dir = self.save_dir / 'visualizations'
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history = {
            'box_loss': [],
            'pose_loss': [],
            'kobj_loss': [],
            'cls_loss': [],
            'dfl_loss': [],
            'mAP50': [],
            'mAP50-95': [],
        }
        
        print(f"📊 Visualization directory: {self.vis_dir}")
    
    def save_training_batch_visualization(self, model, dataset_path, epoch, num_samples=4):
        """可視化訓練批次的預測結果"""
        try:
            dataset_path = Path(dataset_path)
            val_images_dir = dataset_path / 'valid' / 'images'
            val_labels_dir = dataset_path / 'valid' / 'labels'
            
            if not val_images_dir.exists():
                return
            
            # 隨機選擇圖片
            image_files = list(val_images_dir.glob('*'))[:num_samples]
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 16))
            axes = axes.flatten()
            
            for idx, img_path in enumerate(image_files):
                if idx >= 4:
                    break
                
                # 讀取圖片
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # 執行預測
                results = model.predict(img_path, verbose=False, save=False)
                
                if len(results) > 0:
                    result = results[0]
                    
                    # 繪製結果
                    annotated = img_rgb.copy()
                    
                    # 繪製 bbox
                    if result.boxes is not None and len(result.boxes) > 0:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            
                            # 繪製框
                            cv2.rectangle(annotated, 
                                        (int(x1), int(y1)), 
                                        (int(x2), int(y2)), 
                                        (0, 255, 0), 2)
                            
                            # 繪製置信度
                            label = f'{conf:.2f}'
                            cv2.putText(annotated, label, 
                                      (int(x1), int(y1)-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 
                                      0.5, (0, 255, 0), 2)
                    
                    # 繪製關鍵點
                    if result.keypoints is not None and len(result.keypoints) > 0:
                        for kpts in result.keypoints:
                            kpts_data = kpts.xy[0].cpu().numpy()  # [num_kpts, 2]
                            
                            if len(kpts_data) >= 2:
                                # 起點 (綠色)
                                start = kpts_data[0]
                                cv2.circle(annotated, 
                                         (int(start[0]), int(start[1])), 
                                         8, (0, 255, 0), -1)
                                cv2.circle(annotated, 
                                         (int(start[0]), int(start[1])), 
                                         10, (0, 255, 0), 2)
                                
                                # 終點 (藍色)
                                end = kpts_data[1]
                                cv2.circle(annotated, 
                                         (int(end[0]), int(end[1])), 
                                         8, (255, 0, 0), -1)
                                cv2.circle(annotated, 
                                         (int(end[0]), int(end[1])), 
                                         10, (255, 0, 0), 2)
                                
                                # 連接線 (黃色)
                                cv2.line(annotated,
                                       (int(start[0]), int(start[1])),
                                       (int(end[0]), int(end[1])),
                                       (255, 255, 0), 3)
                
                # 顯示在子圖
                axes[idx].imshow(annotated)
                axes[idx].set_title(f'Epoch {epoch} - {img_path.name}')
                axes[idx].axis('off')
            
            # 保存圖片
            plt.tight_layout()
            save_path = self.vis_dir / f'epoch_{epoch:03d}_predictions.jpg'
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"  📸 Saved visualization: {save_path.name}")
            
        except Exception as e:
            print(f"  ⚠️  Visualization error: {e}")
    
    def plot_metrics(self, metrics_csv):
        """繪製訓練指標曲線"""
        try:
            if not metrics_csv.exists():
                return
            
            import pandas as pd
            df = pd.read_csv(metrics_csv)
            
            # 創建2x2子圖
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Loss 曲線
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
            
            # mAP 曲線
            ax = axes[0, 1]
            if 'metrics/mAP50(B)' in df.columns:
                ax.plot(df['epoch'], df['metrics/mAP50(B)'], 
                       label='mAP50 (Box)', linewidth=2, marker='o')
            if 'metrics/mAP50-95(B)' in df.columns:
                ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], 
                       label='mAP50-95 (Box)', linewidth=2, marker='s')
            if 'metrics/mAP50(P)' in df.columns:
                ax.plot(df['epoch'], df['metrics/mAP50(P)'], 
                       label='mAP50 (Pose)', linewidth=2, marker='^')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP')
            ax.set_title('Validation mAP')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 學習率曲線
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
            
            # Precision & Recall
            ax = axes[1, 1]
            if 'metrics/precision(B)' in df.columns:
                ax.plot(df['epoch'], df['metrics/precision(B)'], 
                       label='Precision (Box)', linewidth=2)
            if 'metrics/recall(B)' in df.columns:
                ax.plot(df['epoch'], df['metrics/recall(B)'], 
                       label='Recall (Box)', linewidth=2)
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


def verify_pose_labels(dataset_path):
    """驗證 pose 標註文件"""
    print("\n" + "="*60)
    print("Verifying Pose Label Files")
    print("="*60)
    
    dataset_path = Path(dataset_path)
    
    for split in ['train', 'valid']:
        labels_dir = dataset_path / split / 'labels'
        
        if not labels_dir.exists():
            print(f"❌ {split} labels directory not found")
            return False
        
        # 检查所有 .txt 文件（不再专门找 _pose.txt）
        label_files = list(labels_dir.glob('*.txt'))
        
        if len(label_files) == 0:
            print(f"❌ No label files found in {split}/labels")
            return False
        
        print(f"✅ {split}: Found {len(label_files)} label files")
        
        # 检查格式 - 验证是否为 pose 格式（11列）
        valid_pose_count = 0
        invalid_count = 0
        
        for label_file in label_files[:5]:  # 检查前5个
            with open(label_file, 'r') as f:
                first_line = f.readline().strip()
                if first_line:
                    parts = first_line.split()
                    
                    if len(parts) == 11:
                        valid_pose_count += 1
                    else:
                        invalid_count += 1
                        print(f"   ⚠️  {label_file.name}: {len(parts)} columns (expected 11)")
        
        if invalid_count > 0:
            print(f"   ❌ Found {invalid_count} invalid format files")
            return False
        
        print(f"   ✅ Format verified: 11 columns (pose format)")
    
    return True


def create_pose_dataset_yaml(dataset_path, output_yaml):
    """創建 YOLOv11 pose 專用的 dataset.yaml"""
    
    dataset_path = Path(dataset_path)
    
    train_images = len(list((dataset_path / 'train' / 'images').glob('*')))
    valid_images = len(list((dataset_path / 'valid' / 'images').glob('*')))
    train_pose_labels = len(list((dataset_path / 'train' / 'labels').glob('*_pose.txt')))
    valid_pose_labels = len(list((dataset_path / 'valid' / 'labels').glob('*_pose.txt')))
    
    config = {
        'path': str(dataset_path.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'nc': 1,
        'names': ['pointer'],
        'kpt_shape': [2, 3],  # 2 keypoints, 3 dimensions each (x, y, visibility)
    }
    
    with open(output_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print("\n" + "="*60)
    print("Dataset Configuration")
    print("="*60)
    print(f"Path: {dataset_path}")
    print(f"Train images: {train_images} (pose labels: {train_pose_labels})")
    print(f"Valid images: {valid_images} (pose labels: {valid_pose_labels})")
    print(f"Classes: 1 (pointer)")
    print(f"Keypoints: 2 (start, end)")
    print(f"Config saved: {output_yaml}")
    
    return output_yaml


def train_yolov11_pose():
    """訓練 YOLOv11-pose 模型"""
    
    dataset_path = Path("/home/itemhsu/amtk/gauge/yolo_dataset")
    dataset_yaml = dataset_path / "dataset_pose.yaml"
    
    # 驗證 pose 標註
    if not verify_pose_labels(dataset_path):
        print("\n❌ Pose label verification failed")
        print("Please run: python batch_generate_pose_labels.py")
        return False
    
    # 創建 dataset.yaml
    create_pose_dataset_yaml(dataset_path, dataset_yaml)
    
    # 訓練配置
    config = {
        'data': str(dataset_yaml),
        'epochs': 100,
        'imgsz': 960,
        'batch': 16,
        
        # 學習率
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Warmup
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss 權重
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,      # 高權重確保關鍵點準確
        'kobj': 2.0,       # 關鍵點置信度
        
        # 數據增強（關閉翻轉以保持方向性）
        'fliplr': 0.0,
        'flipud': 0.0,
        'degrees': 0.0,    # 不旋轉
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'mosaic': 1.0,
        
        # 其他設置
        'val': True,
        'plots': True,
        'save': True,
        'save_period': 5,   # 每5個epoch保存一次
        'device': '',
        'workers': 8,
        'project': 'runs/pose',
        'name': 'yolov11_pointer_pose',
        'exist_ok': False,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 0,
        'deterministic': True,
        'amp': True,
        'close_mosaic': 10,
    }
    
    try:
        print("\n" + "="*60)
        print("Initializing YOLOv11-Pose Model")
        print("="*60)
        
        # 嘗試加載 YOLOv11-pose 模型
        model_variants = [
            'yolo11s-pose.pt',
        ]
        
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
                continue
        
        if model is None:
            print("\n📥 Downloading YOLOv11n-pose...")
            try:
                model = YOLO('yolo11n-pose.pt')
                loaded_variant = 'yolo11n-pose.pt'
                print("✅ Downloaded yolo11n-pose.pt")
            except:
                print("📥 Trying YOLOv8n-pose as fallback...")
                model = YOLO('yolov8n-pose.pt')
                loaded_variant = 'yolov8n-pose.pt'
                print("✅ Downloaded yolov8n-pose.pt")
        
        print("\n" + "="*60)
        print("Training Configuration")
        print("="*60)
        print(f"Model: {loaded_variant}")
        print(f"Dataset: {dataset_yaml}")
        print(f"Epochs: {config['epochs']}")
        print(f"Batch size: {config['batch']}")
        print(f"Image size: {config['imgsz']}")
        print(f"Pose loss weight: {config['pose']}")
        print(f"Keypoint obj weight: {config['kobj']}")
        print(f"Augmentation: fliplr={config['fliplr']}, flipud={config['flipud']}")
        print("="*60)
        
        # 創建可視化器
        save_dir = Path('runs/pose/yolov11_pointer_pose')
        visualizer = TrainingVisualizer(save_dir)
        
        # 開始訓練
        print("\n🚀 Starting training...\n")
        
        # 添加回調函數進行可視化
        def on_train_epoch_end(trainer):
            """每個 epoch 結束時的回調"""
            epoch = trainer.epoch
            
            # 每5個epoch可視化一次
            if epoch % 5 == 0 or epoch == trainer.epochs - 1:
                print(f"\n📊 Generating visualizations for epoch {epoch}...")
                visualizer.save_training_batch_visualization(
                    trainer.model, 
                    dataset_path, 
                    epoch
                )
        
        # 訓練模型
        results = model.train(**config)
        
        # 訓練完成後繪製指標
        print("\n📈 Generating final metrics plots...")
        results_dir = Path(results.save_dir)
        metrics_csv = results_dir / 'results.csv'
        
        if metrics_csv.exists():
            visualizer.plot_metrics(metrics_csv)
        
        # 在驗證集上可視化最終結果
        print("\n🎯 Generating final predictions visualization...")
        best_model = YOLO(results_dir / 'weights' / 'best.pt')
        visualizer.save_training_batch_visualization(
            best_model, 
            dataset_path, 
            'final',
            num_samples=8
        )
        
        print("\n" + "="*60)
        print("✅ Training completed successfully!")
        print("="*60)
        print(f"📁 Results directory: {results.save_dir}")
        print(f"🏆 Best weights: {results.save_dir}/weights/best.pt")
        print(f"📝 Last weights: {results.save_dir}/weights/last.pt")
        print(f"📊 Visualizations: {visualizer.vis_dir}")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函數"""
    print("="*60)
    print("YOLOv11-Pose Training for Dial Gauge Pointer Detection")
    print("Training: BBox + 2 Keypoints (start, end)")
    print("With Training Visualization")
    print("="*60)
    
    dataset_path = Path("/home/itemhsu/amtk/gauge/yolo_dataset")
    
    if not dataset_path.exists():
        print(f"❌ Error: Dataset not found at {dataset_path}")
        return
    
    # 檢查數據集結構
    required_dirs = ['train/images', 'train/labels', 'valid/images', 'valid/labels']
    missing_dirs = [d for d in required_dirs 
                   if not (dataset_path / d).exists()]
    
    if missing_dirs:
        print("❌ Error: Required directories not found:")
        for d in missing_dirs:
            print(f"   - {d}")
        return
    
    print("✅ Dataset structure verified")
    
    # 統計數據
    train_images = len(list((dataset_path / 'train/images').glob('*')))
    valid_images = len(list((dataset_path / 'valid/images').glob('*')))
    train_labels = len(list((dataset_path / 'train/labels').glob('*.txt')))
    valid_labels = len(list((dataset_path / 'valid/labels').glob('*.txt')))
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Training: {train_images} images, {train_labels} labels")
    print(f"  Validation: {valid_images} images, {valid_labels} labels")
    
    if train_labels == 0:
        print("\n❌ No labels found!")
        print("Run: python batch_generate_pose_labels.py")
        return
    
    # 檢查第一個標註文件的格式
    sample_label = list((dataset_path / 'train/labels').glob('*.txt'))[0]
    with open(sample_label, 'r') as f:
        first_line = f.readline().strip()
        if first_line:
            parts = first_line.split()
            if len(parts) != 11:
                print(f"\n❌ 標註格式錯誤！")
                print(f"  期望: 11列 (pose格式)")
                print(f"  實際: {len(parts)}列")
                print(f"  示例: {sample_label.name}")
                return
            else:
                print(f"  ✅ 標註格式驗證通過 (11列 pose格式)")
    
    # 開始訓練
    success = train_yolov11_pose()
    
    if success:
        print("\n" + "="*60)
        print("🎉 Training completed successfully!")
        print("="*60)
        print("\n📂 Output files:")
        print("  - runs/pose/yolov11_pointer_pose/weights/best.pt")
        print("  - runs/pose/yolov11_pointer_pose/weights/last.pt")
        print("  - runs/pose/yolov11_pointer_pose/visualizations/*.jpg")
        print("\n🔍 To use the trained model:")
        print("  from ultralytics import YOLO")
        print("  model = YOLO('runs/pose/yolov11_pointer_pose/weights/best.pt')")
        print("  results = model.predict('test_image.jpg')")
        print("="*60)


if __name__ == "__main__":
    main()
