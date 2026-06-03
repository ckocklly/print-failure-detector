import cv2
import numpy as np
import os

# === Folder paths ===
input_folder = "img"
output_folder = "output-diff"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# === Loop through all images ===
for filename in os.listdir(input_folder):
    
    # Only process image files
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 1. Load image
        img = cv2.imread(input_path)

        # 2. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Estimate background
        background = cv2.GaussianBlur(gray, (31,31), 0)
        diff = cv2.absdiff(gray, background)

        # 4. Threshold
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)

        # 5. Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 6. Loop through contours
        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # 7. Save processed image
        cv2.imwrite(output_path, img)

print("Processing complete")