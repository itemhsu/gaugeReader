#!/usr/bin/env python3
"""
指針姿態檢測器 - 檢測表頭中的主要指針並標註起點和終點
輸出YOLO pose格式的標註文件和可視化結果
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import math

class NeedlePoseDetector:
    def __init__(self):
        self.class_name = "needle"  # 只有一個類別：指針
        self.class_id = 0
        
    def detect_circles(self, image):
        """檢測圓形表盤"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用HoughCircles檢測圓形
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=50,
            param2=30,
            minRadius=50,
            maxRadius=300
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return circles
        return None
    
    def detect_lines(self, image, mask=None):
        """檢測直線（可能的指針）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if mask is not None:
            gray = cv2.bitwise_and(gray, mask)
        
        # 邊緣檢測
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 使用HoughLinesP檢測直線
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=10
        )
        
        return lines
    
    def filter_needle_lines(self, lines, center, min_length=40):
        """過濾出可能的指針線段"""
        if lines is None:
            return []
        
        needle_candidates = []
        cx, cy = center
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # 計算線段長度
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length < min_length:
                continue
            
            # 計算線段中點到圓心的距離
            mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
            dist_to_center = np.sqrt((mid_x-cx)**2 + (mid_y-cy)**2)
            
            # 檢查線段是否通過或接近圓心
            # 計算點到直線的距離
            A = y2 - y1
            B = x1 - x2
            C = x2*y1 - x1*y2
            
            if A == 0 and B == 0:
                continue
                
            dist_center_to_line = abs(A*cx + B*cy + C) / np.sqrt(A**2 + B**2)
            
            # 如果線段通過圓心附近，認為是指針候選
            if dist_center_to_line < 20:  # 允許一定的誤差
                needle_candidates.append({
                    'line': line[0],
                    'length': length,
                    'dist_to_center': dist_center_to_line
                })
        
        # 按長度排序，選擇最長的幾條作為主要指針
        needle_candidates.sort(key=lambda x: x['length'], reverse=True)
        
        return needle_candidates
    
    def get_needle_keypoints(self, line, center):
        """獲取指針的關鍵點（起點和終點）"""
        x1, y1, x2, y2 = line
        cx, cy = center
        
        # 計算兩個端點到圓心的距離
        dist1 = np.sqrt((x1-cx)**2 + (y1-cy)**2)
        dist2 = np.sqrt((x2-cx)**2 + (y2-cy)**2)
        
        # 距離圓心近的點作為起點，遠的點作為終點
        if dist1 < dist2:
            start_point = (x1, y1)
            end_point = (x2, y2)
        else:
            start_point = (x2, y2)
            end_point = (x1, y1)
        
        return start_point, end_point
    
    def detect_needles(self, image_path):
        """主要的指針檢測函數"""
        # 讀取圖片
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"無法讀取圖片: {image_path}")
        
        height, width = image.shape[:2]
        
        # 檢測圓形表盤
        circles = self.detect_circles(image)
        
        results = []
        
        if circles is not None:
            for (x, y, r) in circles:
                # 創建圓形遮罩
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                
                # 在表盤區域內檢測直線
                lines = self.detect_lines(image, mask)
                
                # 過濾出指針候選
                needle_candidates = self.filter_needle_lines(lines, (x, y))
                
                # 選擇最主要的指針（最長的幾條）
                main_needles = needle_candidates[:3]  # 最多選擇3條主要指針
                
                for needle in main_needles:
                    line = needle['line']
                    start_point, end_point = self.get_needle_keypoints(line, (x, y))
                    
                    # 計算bounding box
                    x1, y1 = start_point
                    x2, y2 = end_point
                    
                    # 擴展bounding box以包含整個指針
                    margin = 10
                    bbox_x1 = max(0, min(x1, x2) - margin)
                    bbox_y1 = max(0, min(y1, y2) - margin)
                    bbox_x2 = min(width, max(x1, x2) + margin)
                    bbox_y2 = min(height, max(y1, y2) + margin)
                    
                    bbox_width = bbox_x2 - bbox_x1
                    bbox_height = bbox_y2 - bbox_y1
                    bbox_center_x = bbox_x1 + bbox_width / 2
                    bbox_center_y = bbox_y1 + bbox_height / 2
                    
                    # 轉換為YOLO格式（歸一化坐標）
                    yolo_center_x = bbox_center_x / width
                    yolo_center_y = bbox_center_y / height
                    yolo_width = bbox_width / width
                    yolo_height = bbox_height / height
                    
                    # 關鍵點坐標（歸一化）
                    kp1_x = x1 / width
                    kp1_y = y1 / height
                    kp2_x = x2 / width
                    kp2_y = y2 / height
                    
                    results.append({
                        'bbox': (yolo_center_x, yolo_center_y, yolo_width, yolo_height),
                        'keypoints': [(kp1_x, kp1_y, 2), (kp2_x, kp2_y, 2)],  # visibility=2表示可見
                        'start_point': start_point,
                        'end_point': end_point,
                        'bbox_pixel': (bbox_x1, bbox_y1, bbox_x2, bbox_y2),
                        'confidence': needle['length'] / 200.0  # 基於長度的置信度
                    })
        
        return results, image
    
    def save_yolo_pose_annotation(self, results, output_path):
        """保存YOLO pose格式的標註文件"""
        with open(output_path, 'w') as f:
            for result in results:
                bbox = result['bbox']
                keypoints = result['keypoints']
                
                # YOLO pose格式: class_id center_x center_y width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v
                line = f"{self.class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
                
                for kp_x, kp_y, kp_v in keypoints:
                    line += f" {kp_x:.6f} {kp_y:.6f} {kp_v}"
                
                f.write(line + "\n")
    
    def visualize_results(self, image, results, output_path):
        """可視化檢測結果"""
        vis_image = image.copy()
        
        for i, result in enumerate(results):
            start_point = result['start_point']
            end_point = result['end_point']
            bbox_pixel = result['bbox_pixel']
            
            # 繪製bounding box
            x1, y1, x2, y2 = bbox_pixel
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 繪製指針線段
            cv2.line(vis_image, 
                    (int(start_point[0]), int(start_point[1])), 
                    (int(end_point[0]), int(end_point[1])), 
                    (0, 0, 255), 3)
            
            # 繪製關鍵點
            cv2.circle(vis_image, (int(start_point[0]), int(start_point[1])), 5, (255, 0, 0), -1)  # 起點-藍色
            cv2.circle(vis_image, (int(end_point[0]), int(end_point[1])), 5, (0, 255, 255), -1)    # 終點-黃色
            
            # 添加標籤
            cv2.putText(vis_image, f"Needle {i+1}", 
                       (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 標註關鍵點
            cv2.putText(vis_image, "Start", 
                       (int(start_point[0])+10, int(start_point[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            cv2.putText(vis_image, "End", 
                       (int(end_point[0])+10, int(end_point[1])), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # 保存可視化結果
        cv2.imwrite(str(output_path), vis_image)
        
        return vis_image

def main():
    parser = argparse.ArgumentParser(description='指針姿態檢測器')
    parser.add_argument('--input', '-i', 
                       default='/home/itemhsu/amtk/gauge/yolo_dataset/train/images/CIMG2046_jpg.rf.78249a5e17472d875dde93b74d66779c.jpg',
                       help='輸入圖片路徑')
    parser.add_argument('--output_dir', '-o', 
                       default='/home/itemhsu/amtk/gauge/needle_pose_results',
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 初始化檢測器
    detector = NeedlePoseDetector()
    
    # 檢測指針
    print(f"正在處理圖片: {args.input}")
    try:
        results, image = detector.detect_needles(args.input)
        
        if not results:
            print("未檢測到指針")
            return
        
        print(f"檢測到 {len(results)} 個指針")
        
        # 生成輸出文件名
        input_path = Path(args.input)
        base_name = input_path.stem
        
        # 保存YOLO pose標註文件
        annotation_path = output_dir / f"{base_name}.txt"
        detector.save_yolo_pose_annotation(results, annotation_path)
        print(f"標註文件已保存: {annotation_path}")
        
        # 保存可視化結果
        vis_path = output_dir / f"{base_name}_annotated.jpg"
        vis_image = detector.visualize_results(image, results, vis_path)
        print(f"可視化結果已保存: {vis_path}")
        
        # 打印檢測結果詳情
        print("\n檢測結果詳情:")
        print("=" * 50)
        for i, result in enumerate(results):
            bbox = result['bbox']
            keypoints = result['keypoints']
            start_point = result['start_point']
            end_point = result['end_point']
            
            print(f"指針 {i+1}:")
            print(f"  Bounding Box (YOLO格式): {bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f}")
            print(f"  起點坐標 (像素): ({int(start_point[0])}, {int(start_point[1])})")
            print(f"  終點坐標 (像素): ({int(end_point[0])}, {int(end_point[1])})")
            print(f"  起點坐標 (歸一化): ({keypoints[0][0]:.4f}, {keypoints[0][1]:.4f})")
            print(f"  終點坐標 (歸一化): ({keypoints[1][0]:.4f}, {keypoints[1][1]:.4f})")
            print()
        
        # 顯示YOLO pose格式的完整標註
        print("YOLO Pose格式標註:")
        print("=" * 50)
        with open(annotation_path, 'r') as f:
            content = f.read()
            print(content)
            
    except Exception as e:
        print(f"處理過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()