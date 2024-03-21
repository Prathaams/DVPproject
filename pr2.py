from flask import Flask, render_template, Response
import cv2
import numpy as np
from collections import deque
import time
from util import get_parking_spots_bboxes, empty_or_not


app = Flask(__name__)

# Load the mask for parking spots
mask = cv2.imread('mask.png', 0)
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = get_parking_spots_bboxes(connected_components)

# Open the video stream
cap = cv2.VideoCapture('pcl.mp4')

# Dummy data for car orders
car_orders = deque(maxlen=60)  # Store car orders for the last 60 minutes

# Initialize car statistics
car_statistics = {
    'Green Boxes': 0,
    'Red Boxes': 0
}

lot = [
    {
        'model': 'SUV',
        'plate': 'MH-12-DVP',
        'time': '15:29 20 March'
    },
    # Add more blacklist entries
]

# Function to generate frames for the video feed
def gen_frames():
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame for parking spot detection
        green_boxes = 0
        red_boxes = 0
        for spot in spots:
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)

            if spot_status:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                green_boxes += 1
            else:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)
                red_boxes += 1

        car_statistics['Green Boxes'] = green_boxes
        car_statistics['Red Boxes'] = red_boxes

        current_time = time.time()
        if current_time - start_time >= 60:  # Update car orders every minute
            car_orders.append(red_boxes)
            start_time = current_time

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html', car_orders=list(car_orders), car_statistics=car_statistics, lot=lot)

# Route for the video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)