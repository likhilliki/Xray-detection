import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# ------------------ LOAD MODEL ------------------
model = YOLO("best.pt")  # make sure best.pt is in same folder as this script

# ------------------ LOAD IMAGE (with Pillow) ------------------
image_path = r"C:\Users\LIKHIL\foreign_body_detector\test_xray.jpg"  # change if needed

if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image not found at: {image_path}")

# Load with Pillow (handles more formats than OpenCV)
pil_img = Image.open(image_path).convert("RGB")
img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# ------------------ PREDICTION ------------------
results = model.predict(source=img, conf=0.4)

# Draw detections
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = f"{model.names[cls]} ({conf:.2f})"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# ------------------ SAVE OUTPUT ------------------
output_path = r"C:\Users\LIKHIL\foreign_body_detector\output_xray.jpg"
cv2.imwrite(output_path, img)
print(f"✅ Processed image saved at: {output_path}")

# ------------------ SHOW RESULT ------------------
try:
    # If GUI available (local PC), show with OpenCV
    cv2.imshow("Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    # If in Colab/Jupyter (no GUI), show with matplotlib
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
