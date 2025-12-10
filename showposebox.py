import cv2

# 檔案路徑（如果檔案不在同一資料夾，請改成實際路徑）
# 可以視需要切換不同影像／標註檔
# img_path = "CIMG2046_jpg.rf.7de0e4ff95595741b398b5264d817ee7.jpg"
# label_path = "CIMG2046_jpg.rf.7de0e4ff95595741b398b5264d817ee7_pose.txt"

img_path = "CIMG3629_jpg.rf.065630db5a6f777b1ff5b1ad31c95f7a.jpg"
label_path = "CIMG3629_jpg.rf.065630db5a6f777b1ff5b1ad31c95f7a_pose.txt"

# 讀取圖片
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"無法讀取圖片：{img_path}")

img_h, img_w = img.shape[:2]

# 讀取 YOLO 標註檔並畫框 + pose 點
with open(label_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        # 至少要有 cls cx cy w h
        if len(parts) < 5:
            continue

        # YOLO bbox 格式: class cx cy w h（0~1）
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])

        # bbox 轉像素座標
        cx_pixel = cx * img_w
        cy_pixel = cy * img_h
        bw_pixel = bw * img_w
        bh_pixel = bh * img_h

        x1 = int(cx_pixel - bw_pixel / 2)
        y1 = int(cy_pixel - bh_pixel / 2)
        x2 = int(cx_pixel + bw_pixel / 2)
        y2 = int(cy_pixel + bh_pixel / 2)

        # 畫紅色 1 像素框 (OpenCV 預設 BGR，所以紅色是 (0, 0, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)

        # ====== 如果是 YOLO pose 格式，畫出前兩個 keypoints ======
        # 假設格式: class cx cy w h kp1x kp1y v1 kp2x kp2y v2 ...
        if len(parts) >= 11:
            kp1x = float(parts[5])  # 第 1 個點 x (0~1)
            kp1y = float(parts[6])  # 第 1 個點 y (0~1)
            v1   = int(parts[7])    # visibility
            kp2x = float(parts[8])  # 第 2 個點 x (0~1)
            kp2y = float(parts[9])  # 第 2 個點 y (0~1)
            v2   = int(parts[10])

            # 轉成像素座標
            p1 = (int(kp1x * img_w), int(kp1y * img_h))
            p2 = (int(kp2x * img_w), int(kp2y * img_h))

            # 半徑 1 pixel -> 直徑約 2 pixel
            radius = 1

            # v>0 視為可見才畫
            if v1 > 0:
                # 第 1 個 pose 點：藍色點 (BGR: 255,0,0)
                cv2.circle(img, p1, radius, (255, 0, 0), -1)

            if v2 > 0:
                # 第 2 個 pose 點：紅色點 (BGR: 0,0,255)
                cv2.circle(img, p2, radius, (0, 0, 255), -1)

# 顯示結果
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

