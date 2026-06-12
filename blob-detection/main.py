import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import os
import cv2

def identify_holes(img, img_mask=None):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, img_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.THRESH_OTSU solves the shadowing problem
    
    h, w = img_gray.shape
    if img_mask is not None:
        img_mask = cv2.resize(img_mask, (w, h), interpolation=cv2.INTER_AREA)
        img_thresh_masked = cv2.bitwise_or(img_thresh, img_mask)
    else:
        img_thresh_masked = img_thresh

    contours, _ = cv2.findContours(img_thresh_masked, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    img_labeled = img.copy()
    for cnt in contours:
        # Rule out tiny contours (possibly noise)
        area = cv2.contourArea(cnt)
        if area <= 20:
            continue

        rRect = cv2.minAreaRect(cnt)
        center, size, angle = rRect

        # Rule out the frame
        if la.norm(np.array(size) - np.array(img_gray.shape)) / \
           la.norm(np.array(img_gray.shape)) < 0.05:
            continue
        # Rule out near-edge tension wear
        if max(size[0]/size[1], size[1]/size[0]) > 10:
            continue
        
        # Rule out near-corner tension wear
        hull = cv2.convexHull(cnt)
        solidity = area / (cv2.contourArea(hull) + 1e-5)
        if solidity <= 0.6:
            continue

        box = np.int_(cv2.boxPoints(rRect))
        cv2.polylines(img_labeled, [box], True, (0,0,255), 2)
        # miny = np.max(box[:, 1])
        # org = (int(round(center[0])), int(miny))
        # cv2.putText(img_labeled, "hole", org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img_labeled


input_folder = "img"
filename_wo_exts = [
    "test",
    "training (1)"
]
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

for filename_wo_ext in filename_wo_exts:
    filename = filename_wo_ext + ".jpg"
    maskname = filename_wo_ext + "-mask.jpg"
    img = cv2.imread(os.path.join(input_folder, filename))
    img_mask = cv2.imread(os.path.join(input_folder, maskname), 0)
    img_labeled = identify_holes(img, img_mask)
    output_path = os.path.join(output_folder, f'output-{filename}')
    cv2.imwrite(output_path, img_labeled)

print(f'Images have been written to the folder \"{output_folder}\"')
