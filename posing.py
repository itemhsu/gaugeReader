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

def find_line_endpoints(roi):
    """在 ROI 裡找出指針線段 (使用 Canny + HoughLinesP)，回傳 (x1,y1,x2,y2)"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 30, 80, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=10,       # 邊緣點投票門檻
        minLineLength=5,    # 線段最短長度
        maxLineGap=3        # 線段最大中斷長度
    )

    if lines is None:
        return None

    # 選擇最長的那一條線段當作指針
    best = None
    max_len = 0
    for x1, y1, x2, y2 in lines[:, 0]:
        length = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if length > max_len:
            max_len = length
            best = (x1, y1, x2, y2)

    return best

def find_circle_center(roi):
    """在較大的 ROI 裡用 HoughCircles 找錶盤圓心，回傳 (cx, cy, r)，座標是 ROI 內像素"""
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
        maxRadius=60
    )

    if circles is None:
        return None

    circles = np.round(circles[0, :]).astype(int)
    h, w = gray.shape
    roi_center = np.array([w / 2.0, h / 2.0])

    # 選擇最靠近 ROI 中心的那一顆圓
    best = None
    best_d = 1e9
    for x, y, r in circles:
        d = (x - roi_center[0]) ** 2 + (y - roi_center[1]) ** 2
        if d < best_d:
            best_d = d
            best = (x, y, r)

    return best

def get_pointer_keypoints(img, bbox):
    """
    給定整張圖 img 以及一個 bbox (x1,y1,x2,y2)，
    回傳指針起點與終點 (start, end) 兩個 image 座標點。
    """
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = bbox

    # 先在原來 bbox 裡找指針線段
    roi = img[y1:y2, x1:x2]
    line = find_line_endpoints(roi)
    if line is None:
        return None

    lx1, ly1, lx2, ly2 = line

    # 再擴大 bbox，讓圓形錶盤也被包含，方便找圓心
    pad = 40  # 視情況可調整
    ex1 = max(0, x1 - pad)
    ey1 = max(0, y1 - pad)
    ex2 = min(img_w - 1, x2 + pad)
    ey2 = min(img_h - 1, y2 + pad)

    exp_roi = img[ey1:ey2, ex1:ex2]
    circle = find_circle_center(exp_roi)

    # 端點換回整張圖座標
    p1 = np.array([x1 + lx1, y1 + ly1], dtype=float)
    p2 = np.array([x1 + lx2, y1 + ly2], dtype=float)

    if circle is None:
        # 找不到圓心，就直接固定順序 (p1=起點, p2=終點)
        start, end = p1, p2
    else:
        cx_roi, cy_roi, r = circle
        center = np.array([ex1 + cx_roi, ey1 + cy_roi], dtype=float)

        # 離圓心較近者當作起點 (靠近指針基部)
        if np.sum((p1 - center) ** 2) <= np.sum((p2 - center) ** 2):
            start, end = p1, p2
        else:
            start, end = p2, p1

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
            # 若找不到線段，就略過或照原樣輸出（看你需求）
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

# ========= (選擇性) 可視化檢查 =========
# 把起點畫成綠色點，終點畫成藍色點
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

    cv2.circle(debug_img, (int(kp1x), int(kp1y)), 3, (0, 255, 0), -1)  # 起點: 綠
    cv2.circle(debug_img, (int(kp2x), int(kp2y)), 3, (255, 0, 0), -1)  # 終點: 藍

# 顯示結果（在桌面環境下）
cv2.imshow("pose debug", debug_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

