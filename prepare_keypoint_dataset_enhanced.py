#!/usr/bin/env python3
"""
數據準備腳本 (增強版) - 自動檢測指針方向
Enhanced Keypoint Dataset Preparation with Auto Direction Detection
"""
import os
import json
from pathlib import Path
import shutil

def check_bbox_annotations():
    """檢查邊界框標註"""
    
    dataset_path = Path("/home/itemhsu/amtk/gauge/yolo_dataset")
    train_labels = dataset_path / "train" / "labels"
    
    if not train_labels.exists():
        return False, "標註目錄不存在"
    
    # 檢查第一個標註文件
    label_files = list(train_labels.glob("*.txt"))
    if not label_files:
        return False, "沒有標註文件"
    
    # 讀取第一個文件
    with open(label_files[0], 'r') as f:
        first_line = f.readline().strip()
    
    parts = first_line.split()
    
    print(f"\n原始標註格式檢查:")
    print(f"  檔案: {label_files[0].name}")
    print(f"  內容: {first_line}")
    print(f"  欄位數: {len(parts)}")
    
    if len(parts) == 5:
        return True, "邊界框格式 (可以轉換)"
    elif len(parts) > 5 and (len(parts) - 5) % 3 == 0:
        num_keypoints = (len(parts) - 5) // 3
        return False, f"已有關鍵點標註 ({num_keypoints} 個關鍵點)"
    else:
        return False, f"標註格式不正確 ({len(parts)} 個欄位)"

def convert_bbox_to_keypoint_enhanced():
    """
    將邊界框標註轉換為關鍵點標註 (增強版)
    
    策略:
    - 自動檢測指針方向 (垂直 vs 水平)
    - 根據寬高比判斷指針朝向
    """
    
    print("\n" + "=" * 60)
    print("邊界框標註 → 關鍵點標註 轉換 (增強版)")
    print("=" * 60)
    
    dataset_path = Path("/home/itemhsu/amtk/gauge/yolo_dataset")
    
    # 類別定義
    CLASS_NAMES = {
        0: 'dial-photos',
        1: 'center dial',
        2: 'center needle',
        3: 'center origin',
        4: 'left dial',
        5: 'left needle',
        6: 'left origin',
        7: 'right dial',
        8: 'right needle',
        9: 'right origin',
    }
    
    # 關鍵點配置
    KEYPOINT_CONFIG = {
        2: 2, 5: 2, 8: 2,  # needle
        3: 1, 6: 1, 9: 1,  # origin
        1: 1, 4: 1, 7: 1,  # dial
        0: 0,              # photos
    }
    
    print("\n關鍵點配置:")
    print("=" * 60)
    for class_id, num_kpts in KEYPOINT_CONFIG.items():
        if num_kpts > 0:
            print(f"  Class {class_id}: {CLASS_NAMES[class_id]:20s} - {num_kpts} 個關鍵點")
    
    print("\n" + "=" * 60)
    print("⚠️  增強功能說明")
    print("=" * 60)
    print("""
自動方向檢測:
1. 分析邊界框的寬高比 (w/h)
2. 判斷指針朝向:
   - 如果 h > w (高度 > 寬度) → 垂直指針
     • 起點: 底部中心 (中心螺絲)
     • 終點: 頂部中心 (指針尖端)
   
   - 如果 w > h (寬度 > 高度) → 水平指針
     • 起點: 左側中心 (中心螺絲)
     • 終點: 右側中心 (指針尖端)

3. 這樣可以適應不同方向的指針

注意:
- 仍然是估計值,不是真實標註
- 建議用視覺化工具檢查結果
- 如需高精度,請手動標註
    """)
    
    print("=" * 60)
    response = input("\n是否繼續自動轉換? (y/n): ")
    if response.lower() != 'y':
        print("已取消轉換")
        return False
    
    print("\n開始轉換...")
    print("=" * 60)
    
    # 統計指針方向
    vertical_count = 0
    horizontal_count = 0
    
    # 處理 train 和 valid
    for split in ['train', 'valid']:
        labels_dir = dataset_path / split / "labels"
        keypoint_dir = dataset_path / split / "labels_keypoint"
        
        # 創建輸出目錄
        keypoint_dir.mkdir(exist_ok=True)
        
        label_files = list(labels_dir.glob("*.txt"))
        print(f"\n處理 {split} 集: {len(label_files)} 個文件")
        
        converted_count = 0
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) != 5:
                    new_lines.append(line.strip())
                    continue
                
                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                
                num_kpts = KEYPOINT_CONFIG.get(class_id, 0)
                
                if num_kpts == 0:
                    new_line = line.strip()
                
                elif num_kpts == 1:
                    # 1 個關鍵點在中心
                    new_line = f"{class_id} {cx} {cy} {w} {h} {cx} {cy} 2"
                
                elif num_kpts == 2:
                    # 2 個關鍵點 - 自動檢測方向
                    
                    # 判斷指針方向
                    if h > w:
                        # 垂直指針 (高 > 寬)
                        vertical_count += 1
                        # 起點在底部中心
                        kp1_x = cx
                        kp1_y = cy + h / 2
                        # 終點在頂部中心
                        kp2_x = cx
                        kp2_y = cy - h / 2
                    else:
                        # 水平指針 (寬 > 高)
                        horizontal_count += 1
                        # 起點在左側中心
                        kp1_x = cx - w / 2
                        kp1_y = cy
                        # 終點在右側中心
                        kp2_x = cx + w / 2
                        kp2_y = cy
                    
                    kp1_visible = 2
                    kp2_visible = 2
                    
                    new_line = f"{class_id} {cx} {cy} {w} {h} {kp1_x} {kp1_y} {kp1_visible} {kp2_x} {kp2_y} {kp2_visible}"
                
                else:
                    new_line = line.strip()
                
                new_lines.append(new_line)
            
            # 保存到新目錄
            output_file = keypoint_dir / label_file.name
            with open(output_file, 'w') as f:
                f.write('\n'.join(new_lines) + '\n')
            
            converted_count += 1
        
        print(f"  ✅ 已轉換 {converted_count}/{len(label_files)} 個文件")
        print(f"  輸出目錄: {keypoint_dir}")
    
    print("\n" + "=" * 60)
    print("✅ 轉換完成!")
    print("=" * 60)
    
    # 顯示統計
    print(f"\n指針方向統計:")
    print(f"  垂直指針: {vertical_count} 個")
    print(f"  水平指針: {horizontal_count} 個")
    
    if vertical_count > 0 and horizontal_count > 0:
        print(f"\n⚠️  檢測到混合方向的指針")
        print(f"  建議使用視覺化工具檢查轉換結果")
    
    print(f"\n新的關鍵點標註位置:")
    print(f"  Train: {dataset_path}/train/labels_keypoint/")
    print(f"  Valid: {dataset_path}/valid/labels_keypoint/")
    
    return True

def create_keypoint_yaml():
    """創建關鍵點檢測的 dataset.yaml"""
    
    print("\n" + "=" * 60)
    print("創建數據集配置文件")
    print("=" * 60)
    
    yaml_content = """# YOLOv10 Keypoint Detection Dataset Configuration
# 指針關鍵點檢測數據集配置 (增強版 - 自動方向檢測)

path: /home/itemhsu/amtk/gauge/yolo_dataset
train: train/images
val: valid/images

# Classes
names:
  0: dial-photos
  1: center dial
  2: center needle
  3: center origin
  4: left dial
  5: left needle
  6: left origin
  7: right dial
  8: right needle
  9: right origin

# Keypoint Configuration
kpt_shape: [2, 3]

# 關鍵點定義 (自動方向檢測版本):
# needle (2, 5, 8): 2 個關鍵點
#   垂直指針 (h > w):
#     - keypoint 0: 底部中心 (起點/旋轉中心)
#     - keypoint 1: 頂部中心 (終點/指針尖端)
#   水平指針 (w > h):
#     - keypoint 0: 左側中心 (起點/旋轉中心)
#     - keypoint 1: 右側中心 (終點/指針尖端)
# 
# origin (3, 6, 9): 1 個關鍵點
#   - keypoint 0: 中心點
# 
# dial (1, 4, 7): 1 個關鍵點  
#   - keypoint 0: 刻度盤中心

# Keypoint names
keypoint_names:
  - start
  - end

# Flip indices
flip_idx: [1, 0]
"""
    
    yaml_path = Path("/home/itemhsu/amtk/gauge/yolo_dataset/dataset_keypoint.yaml")
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ 已創建配置文件: {yaml_path}")
    
    return yaml_path

def update_labels_symlink():
    """創建符號連結"""
    print("\n" + "=" * 60)
    print("更新標註目錄連結")
    print("=" * 60)
    
    dataset_path = Path("/home/itemhsu/amtk/gauge/yolo_dataset")
    
    for split in ['train', 'valid']:
        labels_dir = dataset_path / split / "labels"
        labels_keypoint_dir = dataset_path / split / "labels_keypoint"
        labels_backup = dataset_path / split / "labels_bbox_backup"
        
        # 備份原始邊界框標註
        if labels_dir.exists() and not labels_backup.exists():
            print(f"\n備份原始標註: {split}/labels → {split}/labels_bbox_backup")
            shutil.copytree(labels_dir, labels_backup)
        
        # 刪除舊的 labels 目錄
        if labels_dir.exists() and not labels_dir.is_symlink():
            shutil.rmtree(labels_dir)
        elif labels_dir.is_symlink():
            labels_dir.unlink()
        
        # 創建符號連結
        if labels_keypoint_dir.exists():
            labels_dir.symlink_to('labels_keypoint')
            print(f"✅ 創建連結: {split}/labels → labels_keypoint")
    
    print("\n現在訓練時會使用關鍵點標註")

def show_statistics():
    """顯示數據集統計信息"""
    
    print("\n" + "=" * 60)
    print("數據集統計")
    print("=" * 60)
    
    dataset_path = Path("/home/itemhsu/amtk/gauge/yolo_dataset")
    
    for split in ['train', 'valid']:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels_keypoint"
        
        if not images_dir.exists() or not labels_dir.exists():
            continue
        
        num_images = len(list(images_dir.glob("*")))
        num_labels = len(list(labels_dir.glob("*.txt")))
        
        print(f"\n{split.upper()} 集:")
        print(f"  圖片數量: {num_images}")
        print(f"  標註數量: {num_labels}")
        
        # 統計關鍵點
        total_objects = 0
        total_keypoints = 0
        class_counts = {}
        
        for label_file in labels_dir.glob("*.txt"):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                total_objects += 1
                
                if len(parts) > 5:
                    num_kpts = (len(parts) - 5) // 3
                    total_keypoints += num_kpts
        
        print(f"  總物體數: {total_objects}")
        print(f"  總關鍵點數: {total_keypoints}")
        
        if class_counts:
            print(f"\n  各類別統計:")
            CLASS_NAMES = {
                0: 'dial-photos', 1: 'center dial', 2: 'center needle',
                3: 'center origin', 4: 'left dial', 5: 'left needle',
                6: 'left origin', 7: 'right dial', 8: 'right needle',
                9: 'right origin'
            }
            for class_id, count in sorted(class_counts.items()):
                name = CLASS_NAMES.get(class_id, f'Class {class_id}')
                print(f"    {name:20s}: {count:4d}")

def main():
    """主函數"""
    
    print("=" * 60)
    print("關鍵點數據準備工具 (增強版)")
    print("Enhanced Keypoint Dataset Preparation")
    print("=" * 60)
    
    # 步驟 1: 檢查原始標註
    print("\n步驟 1: 檢查原始標註格式")
    print("-" * 60)
    
    is_bbox, msg = check_bbox_annotations()
    print(f"檢查結果: {msg}")
    
    if not is_bbox:
        if "已有關鍵點標註" in msg:
            print("\n✅ 數據集已經是關鍵點格式,可以直接訓練!")
            create_keypoint_yaml()
            show_statistics()
            print("\n下一步:")
            print("  python train_yolov10_keypoint_standalone.py")
            return
        else:
            print(f"\n❌ 錯誤: {msg}")
            return
    
    # 步驟 2: 轉換標註 (增強版)
    print("\n步驟 2: 轉換邊界框標註為關鍵點標註 (增強版)")
    print("-" * 60)
    
    success = convert_bbox_to_keypoint_enhanced()
    
    if not success:
        print("\n轉換已取消或失敗")
        return
    
    # 步驟 3: 創建配置文件
    print("\n步驟 3: 創建數據集配置文件")
    print("-" * 60)
    
    yaml_path = create_keypoint_yaml()
    
    # 步驟 4: 更新標註目錄
    print("\n步驟 4: 更新標註目錄")
    print("-" * 60)
    
    response = input("\n是否創建符號連結,使用關鍵點標註? (y/n): ")
    if response.lower() == 'y':
        update_labels_symlink()
    else:
        print("\n⚠️  未創建連結")
    
    # 步驟 5: 顯示統計
    show_statistics()
    
    # 完成
    print("\n" + "=" * 60)
    print("✅ 數據準備完成!")
    print("=" * 60)
    
    print("\n生成的文件:")
    print(f"  1. 關鍵點標註: yolo_dataset/train/labels_keypoint/")
    print(f"  2. 關鍵點標註: yolo_dataset/valid/labels_keypoint/")
    print(f"  3. 配置文件: {yaml_path}")
    print(f"  4. 原始標註備份: yolo_dataset/train/labels_bbox_backup/")
    
    print("\n⚠️  重要: 請使用視覺化工具檢查轉換結果!")
    print("  python visualize_keypoints.py --batch train --num 20")
    
    print("\n下一步:")
    print("  1. 檢查視覺化結果")
    print("  2. 如果滿意,開始訓練:")
    print("     python train_yolov10_keypoint_standalone.py")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
