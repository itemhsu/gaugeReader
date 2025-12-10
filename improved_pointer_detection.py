import cv2
import numpy as np

# ========= 基本設定 =========
img_path   = "CIMG2046_jpg.rf.7de0e4ff95595741b398b5264d817ee7.jpg"
label_path = "CIMG2046_jpg.rf.7de0e4ff95595741b398b5264d817ee7.txt"
out_path   = "CIMG2046_jpg.rf.7de0e4ff95595741b398b5264d817ee7_pose.txt"

# ========= 工具函式 =========
def yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h):
    """YOLO bbox (0~1) -> 左上右下像素座標"""
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return x1, y1, x2, y2

def enhance_pointer_region(roi):
    """增強指針區域，提高對比度"""
    # 轉灰階
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 使用 CLAHE 增強對比度
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # 自適應二值化，更好地分離指針
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # 形態學操作：去除噪點
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return gray, binary

def simple_skeletonize(binary):
    """簡單的骨架化實現（不依賴 opencv-contrib）"""
    # 使用形態學操作進行簡化的骨架化
    skeleton = np.zeros_like(binary)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    temp = binary.copy()
    while True:
        eroded = cv2.erode(temp, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        subset = eroded - opened
        skeleton = cv2.bitwise_or(skeleton, subset)
        temp = eroded.copy()
        
        if cv2.countNonZero(temp) == 0:
            break
    
    return skeleton

def find_line_endpoints_improved(roi):
    """
    改進版指針線段檢測
    結合多種方法：HoughLinesP + 骨架化 + 輪廓分析
    """
    gray, binary = enhance_pointer_region(roi)
    
    # 方法1: 使用骨架化找到指針中心線
    skeleton = simple_skeletonize(binary)
    
    # 方法2: Canny + HoughLinesP (多參數組合)
    edges = cv2.Canny(gray, 20, 100, apertureSize=3)
    
    # 嘗試多組參數
    all_lines = []
    param_sets = [
        {'threshold': 10, 'minLineLength': 8, 'maxLineGap': 3},
        {'threshold': 8, 'minLineLength': 6, 'maxLineGap': 4},
        {'threshold': 12, 'minLineLength': 10, 'maxLineGap': 2},
    ]
    
    for params in param_sets:
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            **params
        )
        if lines is not None:
            all_lines.extend(lines[:, 0].tolist())
    
    # 方法3: 從骨架提取線段
    skeleton_points = np.column_stack(np.where(skeleton > 0))
    if len(skeleton_points) > 5:
        # 使用 PCA 找主軸方向
        mean = np.mean(skeleton_points, axis=0)
        _, eigenvectors = np.linalg.eig(np.cov(skeleton_points.T))
        
        # 找骨架點沿主軸的極值
        principal_axis = eigenvectors[:, 0]
        projections = skeleton_points @ principal_axis
        
        min_idx = np.argmin(projections)
        max_idx = np.argmax(projections)
        
        p1 = skeleton_points[min_idx][::-1]  # (y,x) -> (x,y)
        p2 = skeleton_points[max_idx][::-1]
        
        all_lines.append([int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])])
    
    if not all_lines:
        return None
    
    # 合併和過濾線段
    filtered_lines = merge_similar_lines(all_lines, roi.shape)
    
    # 選擇最佳線段（最長且位置合理）
    best = select_best_line(filtered_lines, roi.shape)
    
    return best

def merge_similar_lines(lines, roi_shape):
    """合併相似的線段"""
    if not lines:
        return []
    
    h, w = roi_shape[:2]
    center = np.array([w / 2, h / 2])
    
    # 計算每條線的特徵：長度、角度、中點位置
    line_features = []
    for x1, y1, x2, y2 in lines:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        angle = np.arctan2(y2 - y1, x2 - x1)
        mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        dist_to_center = np.linalg.norm(mid - center)
        
        line_features.append({
            'line': (x1, y1, x2, y2),
            'length': length,
            'angle': angle,
            'mid': mid,
            'dist': dist_to_center
        })
    
    # 過濾太短的線段
    min_length = max(w, h) * 0.15
    line_features = [lf for lf in line_features if lf['length'] >= min_length]
    
    if not line_features:
        return [max(lines, key=lambda l: np.sqrt((l[2]-l[0])**2 + (l[3]-l[1])**2))]
    
    # 聚類相似的線段（角度和位置相近）
    clusters = []
    angle_threshold = np.pi / 12  # 15度
    
    for lf in line_features:
        merged = False
        for cluster in clusters:
            avg_angle = np.mean([c['angle'] for c in cluster])
            angle_diff = abs(((lf['angle'] - avg_angle + np.pi) % (2 * np.pi)) - np.pi)
            
            if angle_diff < angle_threshold:
                cluster.append(lf)
                merged = True
                break
        
        if not merged:
            clusters.append([lf])
    
    # 每個聚類取最長的線段
    result = []
    for cluster in clusters:
        best = max(cluster, key=lambda x: x['length'])
        result.append(best['line'])
    
    return result

def select_best_line(lines, roi_shape):
    """選擇最佳線段"""
    if not lines:
        return None
    
    h, w = roi_shape[:2]
    center = np.array([w / 2, h / 2])
    
    # 評分函數：長度 + 位置合理性
    best = None
    best_score = -1
    
    for x1, y1, x2, y2 in lines:
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        mid = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        dist_to_center = np.linalg.norm(mid - center)
        
        # 評分：長度為主，靠近中心加分
        max_dist = np.sqrt(w**2 + h**2) / 2
        score = length * (1 + 0.3 * (1 - dist_to_center / max_dist))
        
        if score > best_score:
            best_score = score
            best = (x1, y1, x2, y2)
    
    return best

def find_circle_center_improved(roi):
    """改進版圓心檢測"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 多尺度檢測
    gray_blurred = cv2.medianBlur(gray, 5)
    
    all_circles = []
    param_sets = [
        {'param1': 80, 'param2': 20, 'minRadius': 15, 'maxRadius': 70},
        {'param1': 60, 'param2': 25, 'minRadius': 20, 'maxRadius': 60},
        {'param1': 100, 'param2': 18, 'minRadius': 18, 'maxRadius': 65},
    ]
    
    for params in param_sets:
        circles = cv2.HoughCircles(
            gray_blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=40,
            **params
        )
        if circles is not None:
            all_circles.extend(circles[0, :].tolist())
    
    if not all_circles:
        return None
    
    # 去重和選擇最佳圓
    h, w = gray.shape
    roi_center = np.array([w / 2.0, h / 2.0])
    
    unique_circles = []
    for cx, cy, r in all_circles:
        # 檢查是否與已有圓重複
        is_duplicate = False
        for ux, uy, ur in unique_circles:
            dist = np.sqrt((cx - ux)**2 + (cy - uy)**2)
            if dist < 10 and abs(r - ur) < 5:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_circles.append([cx, cy, r])
    
    # 選擇最靠近ROI中心且大小合理的圓
    best = None
    best_score = 1e9
    
    for x, y, r in unique_circles:
        dist_to_center = (x - roi_center[0])**2 + (y - roi_center[1])**2
        # 偏好中等大小的圓
        size_penalty = abs(r - 35) / 35
        score = dist_to_center * (1 + size_penalty)
        
        if score < best_score:
            best_score = score
            best = (int(x), int(y), int(r))
    
    return best

def refine_endpoints(p1, p2, center, roi, bbox):
    """
    精細化端點位置
    使用圓心信息和圖像梯度來調整端點
    """
    x1, y1, x2, y2 = bbox
    
    # 確定哪個點是起點（靠近圓心）
    if np.linalg.norm(p1 - center) < np.linalg.norm(p2 - center):
        start, end = p1, p2
    else:
        start, end = p2, p1
    
    # 將起點微調到更靠近圓心（指針軸心）
    direction_to_center = center - start
    dist_to_center = np.linalg.norm(direction_to_center)
    
    if dist_to_center > 5:
        direction_to_center = direction_to_center / dist_to_center
        # 將起點向圓心方向移動一小段距離
        start_refined = start + direction_to_center * min(dist_to_center * 0.3, 10)
    else:
        start_refined = center.copy()
    
    # 終點沿指針方向延伸到邊緣
    pointer_direction = end - start
    pointer_length = np.linalg.norm(pointer_direction)
    
    if pointer_length > 0:
        pointer_direction = pointer_direction / pointer_length
        
        # 在ROI中搜索指針尖端（沿方向找邊緣）
        h, w = roi.shape[:2]
        
        # 轉換到ROI座標
        start_roi = start - np.array([x1, y1])
        search_steps = int(max(w, h) * 0.7)
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 沿指針方向搜索
        max_dist = 0
        best_end = end.copy()
        
        for step in range(5, search_steps):
            test_point = start_roi + pointer_direction * step
            tx, ty = int(test_point[0]), int(test_point[1])
            
            if 0 <= tx < w and 0 <= ty < h:
                # 檢查周圍是否有指針像素
                window_size = 3
                x_start = max(0, tx - window_size)
                x_end = min(w, tx + window_size + 1)
                y_start = max(0, ty - window_size)
                y_end = min(h, ty + window_size + 1)
                
                window = binary[y_start:y_end, x_start:x_end]
                if np.sum(window > 0) > 0:
                    max_dist = step
                    best_end = np.array([x1 + tx, y1 + ty])
            else:
                break
        
        if max_dist > pointer_length * 0.8:
            end_refined = best_end
        else:
            end_refined = end
    else:
        end_refined = end
    
    return start_refined, end_refined

def get_pointer_keypoints(img, bbox):
    """
    給定整張圖 img 以及一個 bbox (x1,y1,x2,y2)，
    回傳指針起點與終點 (start, end) 兩個 image 座標點。
    """
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    # 先在原來 bbox 裡找指針線段
    roi = img[y1:y2, x1:x2]
    line = find_line_endpoints_improved(roi)
    if line is None:
        return None

    lx1, ly1, lx2, ly2 = line

    # 擴大 bbox，包含錶盤圓形區域
    pad = 50  # 增加padding
    ex1 = max(0, x1 - pad)
    ey1 = max(0, y1 - pad)
    ex2 = min(img_w - 1, x2 + pad)
    ey2 = min(img_h - 1, y2 + pad)

    exp_roi = img[ey1:ey2, ex1:ex2]
    circle = find_circle_center_improved(exp_roi)

    # 端點換回整張圖座標
    p1 = np.array([x1 + lx1, y1 + ly1], dtype=float)
    p2 = np.array([x1 + lx2, y1 + ly2], dtype=float)

    if circle is None:
        # 找不到圓心，使用bbox中心作為參考
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=float)
        if np.sum((p1 - center) ** 2) <= np.sum((p2 - center) ** 2):
            start, end = p1, p2
        else:
            start, end = p2, p1
    else:
        cx_roi, cy_roi, r = circle
        center = np.array([ex1 + cx_roi, ey1 + cy_roi], dtype=float)
        
        # 精細化端點
        start, end = refine_endpoints(p1, p2, center, roi, bbox)

    return tuple(start), tuple(end)

# ========= 主流程：讀檔、找 keypoints、寫 YOLO pose =========
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"讀不到圖片：{img_path}")

img_h, img_w = img.shape[:2]

pose_lines = []

with open(label_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        cls, cx, cy, bw, bh = map(float, line.split())
        # 原始 bbox 轉成像素座標
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h)

        start_end = get_pointer_keypoints(img, (x1, y1, x2, y2))
        if start_end is None:
            print("Warning: 找不到指針線段，略過：", line)
            continue

        (sx, sy), (ex, ey) = start_end

        # 轉成 YOLO 0~1 座標
        kp1x = sx / img_w
        kp1y = sy / img_h
        kp2x = ex / img_w
        kp2y = ey / img_h

        v1 = 2  # 2=可見且標註
        v2 = 2

        pose_line = f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " \
                    f"{kp1x:.6f} {kp1y:.6f} {v1} " \
                    f"{kp2x:.6f} {kp2y:.6f} {v2}\n"
        pose_lines.append(pose_line)

# 寫出 YOLO pose 標註檔
with open(out_path, "w", encoding="utf-8") as f:
    f.writelines(pose_lines)

print("已輸出 YOLO pose 標註到：", out_path)

# ========= 可視化檢查（增強版）=========
debug_img = img.copy()
for line in pose_lines:
    parts = line.strip().split()
    if len(parts) < 11:
        continue
    _, cx, cy, bw, bh, kp1x, kp1y, v1, kp2x, kp2y, v2 = parts
    kp1x = float(kp1x) * img_w
    kp1y = float(kp1y) * img_h
    kp2x = float(kp2x) * img_w
    kp2y = float(kp2y) * img_h

    # 畫指針線
    cv2.line(debug_img, (int(kp1x), int(kp1y)), (int(kp2x), int(kp2y)), (0, 255, 255), 2)
    
    # 起點: 大綠圈
    cv2.circle(debug_img, (int(kp1x), int(kp1y)), 5, (0, 255, 0), -1)
    cv2.circle(debug_img, (int(kp1x), int(kp1y)), 7, (0, 255, 0), 2)
    
    # 終點: 大藍圈
    cv2.circle(debug_img, (int(kp2x), int(kp2y)), 5, (255, 0, 0), -1)
    cv2.circle(debug_img, (int(kp2x), int(kp2y)), 7, (255, 0, 0), 2)

# 保存可視化結果
cv2.imwrite("debug_pose_visualization.jpg", debug_img)
print("可視化結果已保存到: debug_pose_visualization.jpg")

# 顯示結果（在桌面環境下）
cv2.imshow("pose debug", debug_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
