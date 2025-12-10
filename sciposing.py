import cv2
import numpy as np
from skimage import filters, morphology, exposure

# ========= 基本設定 =========
img_path   = "CIMG2046_jpg.rf.7de0e4ff95595741b398b5264d817ee7.jpg"
label_path = "CIMG2046_jpg.rf.7de0e4ff95595741b398b5264d817ee7.txt"
out_path   = "CIMG2046_jpg.rf.7de0e4ff95595741b398b5264d817ee7_pose.txt"

# ========= 工具函式 =========
def yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h):
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return x1, y1, x2, y2

def find_pointer_with_skeleton(roi):
    """
    以 skeletonize 找 ROI 內的指針，
    回傳骨架上兩個最遠點 (x1,y1,x2,y2)（座標為 ROI 內像素）。
    """
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 對比度拉伸讓指針更明顯
    gray = exposure.rescale_intensity(gray, in_range='image', out_range=(0, 255))
    gray = gray.astype(np.uint8)

    # Otsu 二值化：假設指針較暗
    thresh = filters.threshold_otsu(gray)
    bw = gray < thresh

    # 去除小雜訊
    bw = morphology.remove_small_objects(bw, min_size=30)

    # 骨架化 → 單像素寬線段
    skel = morphology.skeletonize(bw)

    # 找出骨架所有像素座標 (row = y, col = x)
    ys, xs = np.where(skel)
    if len(xs) < 2:
        return None

    coords = np.column_stack((xs, ys))  # (N,2)

    # 在骨架點中選「距離最遠」的兩點，當作線段兩端
    # 為避免 O(N^2) 太慢，如果點非常多可以隨機抽樣；這裡通常 ROI 很小，直接算。
    max_d = -1.0
    p1 = p2 = None
    n = len(coords)
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i, 0] - coords[j, 0]
            dy = coords[i, 1] - coords[j, 1]
            d = dx * dx + dy * dy
            if d > max_d:
                max_d = d
                p1 = coords[i]
                p2 = coords[j]

    if p1 is None or p2 is None:
        return None

    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    return x1, y1, x2, y2

def find_circle_center(roi):
    """用 HoughCircles 找錶盤圓心（座標為 ROI 內像素）。"""
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=80,
        param2=20,
        minRadius=20,
        maxRadius=80
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    h, w = gray.shape
    roi_center = np.array([w / 2.0, h / 2.0])

    best = None
    best_d = 1e9
    for x, y, r in circles:
        d = (x - roi_center[0]) ** 2 + (y - roi_center[1]) ** 2
        if d < best_d:
            best_d = d
            best = (x, y, r)

    return best  # (cx, cy, r)

def get_pointer_keypoints(img, bbox):
    """
    給定整張圖 img 以及一個 bbox (x1,y1,x2,y2)，
    用 skeletonize 找出指針兩端點，並依圓心決定起點/終點。
    """
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    # 1) 在原 bbox 裡找指針骨架兩端
    roi = img[y1:y2, x1:x2]
    line = find_pointer_with_skeleton(roi)
    if line is None:
        return None
    lx1, ly1, lx2, ly2 = line

    # 2) 擴大 bbox 找圓心
    pad = 40
    ex1 = max(0, x1 - pad)
    ey1 = max(0, y1 - pad)
    ex2 = min(img_w - 1, x2 + pad)
    ey2 = min(img_h - 1, y2 + pad)

    exp_roi = img[ey1:ey2, ex1:ex2]
    circle = find_circle_center(exp_roi)

    # 3) 轉回整張圖座標
    p1 = np.array([x1 + lx1, y1 + ly1], dtype=float)
    p2 = np.array([x1 + lx2, y1 + ly2], dtype=float)

    if circle is None:
        # 找不到圓心，就直接固定 p1=起點, p2=終點
        start, end = p1, p2
    else:
        cx_roi, cy_roi, r = circle
        center = np.array([ex1 + cx_roi, ey1 + cy_roi], dtype=float)

        # 離圓心較近的端點 = 起點，較遠 = 終點
        if np.sum((p1 - center) ** 2) <= np.sum((p2 - center) ** 2):
            start, end = p1, p2
        else:
            start, end = p2, p1

    return tuple(start), tuple(end)

# ========= 主流程：讀 YOLO bbox → 產生 pose =========
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
        # bbox 像素座標
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h)

        start_end = get_pointer_keypoints(img, (x1, y1, x2, y2))
        if start_end is None:
            print("Warning: skeletonize 找不到指針，略過：", line)
            continue

        (sx, sy), (ex, ey) = start_end

        # 轉回 0~1 座標
        kp1x = sx / img_w
        kp1y = sy / img_h
        kp2x = ex / img_w
        kp2y = ey / img_h
        v1, v2 = 2, 2   # 2 = 可見且標註

        pose_line = (
            f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} "
            f"{kp1x:.6f} {kp1y:.6f} {v1} "
            f"{kp2x:.6f} {kp2y:.6f} {v2}\n"
        )
        pose_lines.append(pose_line)

# 寫出 YOLO pose 標註
with open(out_path, "w", encoding="utf-8") as f:
    f.writelines(pose_lines)

print("已輸出 YOLO pose 標註到：", out_path)

# ========= (選擇性) 可視化檢查 =========
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

    cv2.circle(debug_img, (int(kp1x), int(kp1y)), 3, (0, 255, 0), -1)  # 起點：綠
    cv2.circle(debug_img, (int(kp2x), int(kp2y)), 3, (255, 0, 0), -1)  # 終點：藍

# 若在桌面環境:
cv2.imshow("pose debug", debug_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

