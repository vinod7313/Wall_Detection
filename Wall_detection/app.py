
import os
from django.conf import settings
from django.core.wsgi import get_wsgi_application
from django.http import JsonResponse
from django.shortcuts import render
from django.urls import path
from django.core.management import execute_from_command_line
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from PIL import Image
import torch
import io

# Django Setup
BASE_DIR = os.path.dirname(__file__)
settings.configure(
    DEBUG=True,
    SECRET_KEY='devkey',
    ROOT_URLCONF=__name__,
    ALLOWED_HOSTS=['*'],
    MIDDLEWARE=[],
    INSTALLED_APPS=[
        'django.contrib.staticfiles',
    ],
    TEMPLATES=[{
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
    }],
    STATIC_URL='/static/',
)
application = get_wsgi_application()

# Load pre-trained YOLOv5s model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

def calculate_perimeter(boxes):
    """Calculate total perimeter from bounding boxes"""
    perimeter = 0.0
    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        perimeter += 2 * (width + height)
    return perimeter

@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        results = model(image, size=640)
        df = results.pandas().xyxy[0]

        predictions = []
        boxes = []
        for _, row in df.iterrows():
            box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            confidence = float(row['confidence'])
            boxes.append(box)
            predictions.append({
                'bbox': box,
                'confidence': confidence,
                'class': row['name']
            })

        perimeter = calculate_perimeter(boxes)

        response = {
            'predictions': predictions,
            'total_perimeter': perimeter
        }
        return JsonResponse(response, safe=False)

    return render(request, 'upload.html')

urlpatterns = [
    path('', upload_image),
]

# HTML Template
if not os.path.exists('templates'):
    os.makedirs('templates')
with open('templates/upload.html', 'w') as f:
    f.write("""
    <!DOCTYPE html>
    <html>
    <body>
    <h2>Upload Image for Object Detection</h2>
    <form method="post" enctype="multipart/form-data">
      {% csrf_token %}
      <input type="file" name="image"><br><br>
      <input type="submit" value="Upload">
    </form>
    </body>
    </html>
    """)

if __name__ == "__main__":
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', '__main__')
    execute_from_command_line(['manage.py', 'runserver', '0.0.0.0:8000'])
