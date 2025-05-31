# 🧱 Wall Detection using Django + YOLOv5

This project is a Django-based object detection system that uses a pre-trained YOLOv5 model to detect walls (or proxy objects like TVs, couches, etc. from COCO classes) in uploaded images.

---

## 🚀 Features

- Upload an image via a Django web interface.
- Run YOLOv5 (pre-trained) object detection on the image.
- Display detected objects with:
  - Bounding box coordinates
  - Class labels
  - Confidence scores
- Calculate and return the **total perimeter** of all bounding boxes.
- Output is returned in structured JSON format.

---
