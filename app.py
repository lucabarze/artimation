from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
from time import sleep
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large

# MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Load the pre-trained MobileNetV2 model
model = MobileNetV3Large(weights='imagenet')

app = Flask(__name__)
fps = 10  # Reduced frame rate
frame_time = 1. / fps

# Variable to store the current prediction
current_prediction = ""

# Canvas settings
canvas_width, canvas_height = 320, 240  # Reduced canvas size
canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

def euclidean_distance(a, b):
    x1, y1 = a
    x2, y2 = b
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def gen_frames_video():
    camera = cv2.VideoCapture(0)
    global canvas, last_x, last_y

    last_x, last_y = None, None  # variables to store last hand position

    while True:
        sleep(frame_time)  # Introduce delay to lower frame rate
        success, frame = camera.read()  
        if not success:
            break
        else:
            # Resize frame to match canvas, maintaining aspect ratio
            height, width, _ = frame.shape
            new_height = canvas_height
            new_width = int(width * new_height / height)
            frame = cv2.resize(frame, (new_width, new_height))
            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame with MediaPipe Hands
            result = hands.process(frame_rgb)

            # Draw hand landmarks on the frame
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:

                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    h, w, c = frame.shape
                    cx_center, cy_center = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * w), int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * h)


                    for id, lm in enumerate(hand_landmarks.landmark):
                        # Get the hand landmark coordinates
                        cx, cy = int(lm.x * w), int(lm.y * h)

                        # Draw a circle for the index finger tip
                        if id == mp_hands.HandLandmark.INDEX_FINGER_TIP:  # index finger tip
                            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                    # Check if a thumbs-up gesture is being made
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                
                    if thumb_tip.y < index_tip.y and \
                       euclidean_distance((thumb_tip.x * w, thumb_tip.y * h), (cx_center, cy_center)) > 50 and \
                       euclidean_distance((index_tip.x * w, index_tip.y * h), (cx_center, cy_center)) < 50 and \
                       euclidean_distance((middle_tip.x * w, middle_tip.y * h), (cx_center, cy_center)) < 50 and \
                       euclidean_distance((ring_tip.x * w, ring_tip.y * h), (cx_center, cy_center)) < 50 and \
                       euclidean_distance((pinky_tip.x * w, pinky_tip.y * h), (cx_center, cy_center)) < 50:  # the thresholds should be adjusted based on your specific camera and how close you hold your hand to it
                        # Reset the canvas
                        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

                    # Get the coordinates of the index fingertip
                    index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    x, y = int(index_fingertip.x * canvas_width), int(index_fingertip.y * canvas_height)

                    # If there is a last hand position, draw a line from it to the current position
                    if last_x is not None and last_y is not None:
                        cv2.line(canvas, (last_x, last_y), (x, y), (255, 0, 0), 3)
                    last_x, last_y = x, y  # update last hand position

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_frames_canvas():
    global current_prediction
    while True:
        sleep(frame_time)  # Introduce delay to match video frame rate

        # Predict the object drawn on the canvas
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        resized_canvas = cv2.resize(canvas_rgb, (224, 224))  # MobileNetV2 expects input of size (224, 224)
        resized_canvas = np.expand_dims(resized_canvas, axis=0)
        predictions = model.predict(resized_canvas)
        current_prediction = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0][0][1]

        # Encode the canvas in JPEG format
        ret, buffer = cv2.imencode('.jpg', canvas)
        canvas_frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + canvas_frame + b'\r\n')

@app.route('/reset_canvas', methods=['POST'])
def reset_canvas():
    global canvas
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    return ('', 204)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/canvas_feed')
def canvas_feed():
    return Response(gen_frames_canvas(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/prediction')
def prediction():
    global current_prediction
    return jsonify(prediction=current_prediction)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True)
