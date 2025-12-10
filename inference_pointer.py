#!/usr/bin/env python3
"""
YOLOv11-Pose Inference and Visualization
对指针进行检测并可视化结果
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO


def draw_pointer_results(image, results, line_thickness=1, point_size=2):
    """
    在图像上绘制检测结果
    
    Args:
        image: 输入图像
        results: YOLO 检测结果
        line_thickness: 线条粗细 (红色矩形)
        point_size: 关键点大小
    
    Returns:
        annotated_image: 标注后的图像
    """
    annotated = image.copy()
    
    if len(results) == 0:
        return annotated
    
    result = results[0]
    
    # 1. 绘制边界框 (红色，1px)
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # 绘制红色矩形框 (1px)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), line_thickness)
    
    # 2. 绘制关键点和连线
    if result.keypoints is not None and len(result.keypoints) > 0:
        for kpts in result.keypoints:
            # 获取关键点坐标 [num_keypoints, 2]
            kpts_xy = kpts.xy[0].cpu().numpy()
            
            if len(kpts_xy) >= 2:
                # 起点 (绿色，2px)
                start_point = kpts_xy[0].astype(int)
                cv2.circle(
                    annotated,
                    tuple(start_point),
                    point_size,
                    (0, 255, 0),  # 绿色
                    -1  # 填充
                )
                
                # 终点 (蓝色，2px)
                end_point = kpts_xy[1].astype(int)
                cv2.circle(
                    annotated,
                    tuple(end_point),
                    point_size,
                    (255, 0, 0),  # 蓝色
                    -1  # 填充
                )
                
                # 连接起点和终点 (黄色线，1px)
                cv2.line(
                    annotated,
                    tuple(start_point),
                    tuple(end_point),
                    (0, 255, 255),  # 黄色
                    line_thickness
                )
    
    return annotated


def inference_single_image(model_path, image_path, output_path=None, show=False):
    """
    对单张图片进行推理
    
    Args:
        model_path: 模型路径
        image_path: 图片路径
        output_path: 输出路径 (可选)
        show: 是否显示结果
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Processing image: {image_path}")
    image = cv2.imread(str(image_path))
    
    if image is None:
        print(f"❌ Failed to read image: {image_path}")
        return None
    
    # 推理
    results = model.predict(
        image_path,
        imgsz=960,
        conf=0.25,  # 置信度阈值
        verbose=False
    )
    
    # 可视化
    annotated = draw_pointer_results(image, results)
    
    # 保存结果
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"✅ Saved result to: {output_path}")
    
    # 显示结果
    if show:
        cv2.imshow('Pointer Detection', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # 打印检测信息
    if len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            print(f"   Detected {len(result.boxes)} pointer(s)")
            
            if result.keypoints is not None:
                for idx, kpts in enumerate(result.keypoints):
                    kpts_xy = kpts.xy[0].cpu().numpy()
                    if len(kpts_xy) >= 2:
                        print(f"   Pointer {idx+1}:")
                        print(f"     Start: ({kpts_xy[0][0]:.1f}, {kpts_xy[0][1]:.1f})")
                        print(f"     End:   ({kpts_xy[1][0]:.1f}, {kpts_xy[1][1]:.1f})")
    
    return annotated


def inference_directory(model_path, input_dir, output_dir, show=False):
    """
    对目录中的所有图片进行推理
    
    Args:
        model_path: 模型路径
        input_dir: 输入目录
        output_dir: 输出目录
        show: 是否显示每张结果
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    # 获取所有图片
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f'*{ext}'))
        image_files.extend(input_dir.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    print(f"Output directory: {output_dir}")
    print()
    
    # 加载模型
    model = YOLO(model_path)
    
    # 处理每张图片
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"   ❌ Failed to read image")
            continue
        
        # 推理
        results = model.predict(
            image_path,
            conf=0.25,
            verbose=False
        )
        
        # 可视化
        annotated = draw_pointer_results(image, results)
        
        # 保存
        output_path = output_dir / f"{image_path.stem}_detected{image_path.suffix}"
        cv2.imwrite(str(output_path), annotated)
        
        # 打印检测结果
        if len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                print(f"   ✅ Detected {len(result.boxes)} pointer(s)")
        
        # 显示
        if show:
            cv2.imshow('Pointer Detection', annotated)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
    
    if show:
        cv2.destroyAllWindows()
    
    print(f"\n✅ Processing complete!")
    print(f"Results saved to: {output_dir}")


def inference_webcam(model_path):
    """
    使用网络摄像头进行实时推理
    
    Args:
        model_path: 模型路径
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Failed to open webcam")
        return
    
    print("Press 'q' to quit, 's' to save screenshot")
    screenshot_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 推理
        results = model.predict(
            frame,
            conf=0.25,
            verbose=False
        )
        
        # 可视化
        annotated = draw_pointer_results(frame, results)
        
        # 显示 FPS
        cv2.putText(
            annotated,
            "Press 'q' to quit, 's' to save",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # 显示结果
        cv2.imshow('Pointer Detection - Webcam', annotated)
        
        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            screenshot_path = f'screenshot_{screenshot_count:03d}.jpg'
            cv2.imwrite(screenshot_path, annotated)
            print(f"Screenshot saved: {screenshot_path}")
            screenshot_count += 1
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv11-Pose Inference for Pointer Detection'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='runs/pose/yolov11_pointer_pose6/weights/best.pt',
        help='Path to model weights'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Image file, directory, or "webcam"'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='inference_results',
        help='Output directory'
    )
    
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display results'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold'
    )
    
    args = parser.parse_args()
    
    # 检查模型是否存在
    if not Path(args.model).exists():
        print(f"❌ Model not found: {args.model}")
        return
    
    print("="*70)
    print("YOLOv11-Pose Pointer Detection")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print()
    
    # 根据输入类型选择处理方式
    if args.source.lower() == 'webcam':
        # 网络摄像头
        inference_webcam(args.model)
    
    elif Path(args.source).is_file():
        # 单张图片
        output_path = Path(args.output) / f"{Path(args.source).stem}_detected.jpg"
        inference_single_image(
            args.model,
            args.source,
            output_path,
            show=args.show
        )
    
    elif Path(args.source).is_dir():
        # 目录
        inference_directory(
            args.model,
            args.source,
            args.output,
            show=args.show
        )
    
    else:
        print(f"❌ Invalid source: {args.source}")
        print("Source should be:")
        print("  - Image file (e.g., image.jpg)")
        print("  - Directory (e.g., ./test_images)")
        print("  - 'webcam' for real-time detection")


if __name__ == "__main__":
    main()
