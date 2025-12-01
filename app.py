from flask import Flask, request, render_template, session, redirect, url_for
import cv2
import math
import os
import requests
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load YOLO model
yolo_model = YOLO("weights/besthel.pt")
class_labels = ["helmet", "plate", "rider"]

# Ensure directories exist for saving images
output_dir = "static/Media"
os.makedirs(os.path.join(output_dir, "rider_without_helmet"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "plates"), exist_ok=True)

# Plate Recognizer API details
PLATE_RECOGNIZER_URL = 'https://api.platerecognizer.com/v1/plate-reader/'
PLATE_RECOGNIZER_TOKEN = '99cfc569d4249c3bb1c289d92a2dae77af7fe319'
regions = ["mx", "in"]  # Regions for plate detection (e.g., "in" for India)

def recognize_plate(plate_image_path):
    """Calls Plate Recognizer API to extract text from the plate image."""
    with open(plate_image_path, 'rb') as fp:
        response = requests.post(
            PLATE_RECOGNIZER_URL,
            data=dict(regions=regions),  # Optional
            files=dict(upload=fp),
            headers={'Authorization': f'Token {PLATE_RECOGNIZER_TOKEN}'}
        )
    return response.json()

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_path = os.path.join(output_dir, "img.png")
            file.save(image_path)

            # Perform object detection
            img = cv2.imread(image_path)
            results = yolo_model(img)
            rider_boxes, helmet_boxes, plate_boxes = [], [], []
            results_mapped = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    if class_labels[cls] == "rider":
                        rider_boxes.append((x1, y1, x2, y2))
                    elif class_labels[cls] == "helmet":
                        helmet_boxes.append((x1, y1, x2, y2))
                    elif class_labels[cls] == "plate":
                        plate_boxes.append((x1, y1, x2, y2))

            for i, (rx1, ry1, rx2, ry2) in enumerate(rider_boxes):
                rider_has_helmet = any(rx1 < hx1 < rx2 and ry1 < hy1 < ry2 for hx1, hy1, hx2, hy2 in helmet_boxes)
                if not rider_has_helmet:
                    rider_img = img[ry1:ry2, rx1:rx2]
                    rider_filename = f"rider_without_helmet_{i}.jpg"
                    cv2.imwrite(os.path.join(output_dir, "rider_without_helmet", rider_filename), rider_img)

                    closest_plate = min(plate_boxes, key=lambda p: math.dist(((rx1+rx2)//2, (ry1+ry2)//2), ((p[0]+p[2])//2, (p[1]+p[3])//2)), default=None)
                    plate_filename = "No Plate"
                    plate_text = "No Plate Found"
                    if closest_plate:
                        px1, py1, px2, py2 = closest_plate
                        plate_img = img[py1:py2, px1:px2]
                        plate_filename = f"plate_of_rider_{i}.jpg"
                        plate_save_path = os.path.join(output_dir, "plates", plate_filename)
                        cv2.imwrite(plate_save_path, plate_img)

                        # Call the Plate Recognizer API to get the plate number
                        plate_response = recognize_plate(plate_save_path)
                        if plate_response.get('results'):
                            plate_text = plate_response['results'][0]['plate'].upper()


                    results_mapped.append({"rider": rider_filename, "plate": plate_filename, "plate_text": plate_text})

            for box, label, color in zip([rider_boxes, helmet_boxes, plate_boxes], ["Rider", "Helmet", "Plate"], [(0, 255, 0), (255, 0, 0), (0, 0, 255)]):
                for (x1, y1, x2, y2) in box:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            cv2.imwrite(os.path.join(output_dir, "processed_image.png"), img)
            session['results_mapped'] = results_mapped

            return redirect(url_for('display_image'))

    return render_template('index.html')

@app.route('/display_image')
def display_image():
    return render_template('display_image.html', image=url_for('static', filename='Media/processed_image.png'), results=session.get('results_mapped', []))

@app.route('/results')
def results():
    return render_template('results.html', results=session.get('results_mapped', []))

if __name__ == '__main__':
    app.run(debug=True)

