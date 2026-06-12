import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
import cv2

def identify_holes(img, img_mask):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.THRESH_OTSU solves the shadowing problem
    
    h, w = img_gray.shape
    img_mask = cv2.resize(img_mask, (w, h), interpolation=cv2.INTER_AREA)
    img_thresh_masked = cv2.bitwise_or(img_thresh, img_mask)

    contours, _ = cv2.findContours(img_thresh_masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    img_labeled = img.copy()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 20:
            continue

        rRect = cv2.minAreaRect(cnt)
        center, size, angle = rRect
        if la.norm(np.array(size) - np.array(img_gray.shape)) / \
           la.norm(np.array(img_gray.shape)) < 0.05:
            continue
        if max(size[0]/size[1], size[1]/size[0]) > 10:
            continue

        box = np.int_(cv2.boxPoints(rRect))
        cv2.polylines(img_labeled, [box], True, (0,0,255), 2)
        # miny = np.max(box[:, 1])
        # org = (int(round(center[0])), int(miny))
        # cv2.putText(img_labeled, "hole", org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img_labeled


input_folder = "img"
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

img = cv2.imread("img/training (1).jpg")
img_mask = cv2.imread("img/training (1)-mask.jpg", 0)
img_labeled = identify_holes(img, img_mask)
output_path = os.path.join(output_folder, "output.jpg")
cv2.imwrite(output_path, img_labeled)

print(f'Images have been written to the folder \"{output_folder}\"')
