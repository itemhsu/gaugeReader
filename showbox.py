import cv2

# 檔案路徑（如果檔案不在同一資料夾，請改成實際路徑）
img_path = "CIMG2046_jpg.rf.7de0e4ff95595741b398b5264d817ee7.jpg"
label_path = "CIMG2046_jpg.rf.7de0e4ff95595741b398b5264d817ee7.txt"
img_path = "CIMG3629_jpg.rf.065630db5a6f777b1ff5b1ad31c95f7a.jpg"
label_path = "CIMG3629_jpg.rf.065630db5a6f777b1ff5b1ad31c95f7a_pose.txt"

# 讀取圖片
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"無法讀取圖片：{img_path}")

img_h, img_w = img.shape[:2]

# 讀取 YOLO 標註檔並畫框
with open(label_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # YOLO 格式: class cx cy w h（全部是 0~1 的比例）
        parts = line.split()

        if len(parts) < 5:
           continue

        # Read first 5 values only
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:5])
        #if len(parts) != 5:
        #    continue

        #cls_id, cx, cy, bw, bh = map(float, parts)

        # 轉成像素座標
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

# 顯示結果
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

