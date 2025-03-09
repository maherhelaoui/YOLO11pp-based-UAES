#Install Required Libraries
#pip install opencv-python
#pip install torch torchvision
#pip install ultralytics  # For YOLO11++
#pip install pyttsx3  # Text-to-Speech library
#pip install numpy
#pip install threading # Used for parallel collective run


#Import Libraries:

import cv2
import torch
import numpy as np
import pyttsx3
from ultralytics import YOLO
# Load YOLO model 
model = YOLO("yolo11++.pt") 

engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

Capture Video from Camera (or Simulated Feed): Capture the video feed from the car's camera (simulated or real-time).

# Open the camera feed (0 for the default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

#Object Detection Using YOLO: Detect objects such as vehicles, pedestrians, traffic signs, etc., using YOLO.

def detect_objects(frame):
    # Perform inference (object detection)
    results = model(frame)

    # Get results (objects, boxes, and scores)
    detections = results.pandas().xywh[0]  # Convert results to Pandas dataframe
    return detections

#Dynamic Expert System Decision Logic: Based on the objects detected in the video feed, we can create a set of rules to make driving decisions (e.g., stop for a pedestrian, slow down for a car ahead). Then, we provide audio feedback using the TTS system.

def decision_making(detections):
    for index, row in detections.iterrows():
        label = row['name']
        confidence = row['confidence']

        if label == 'person' and confidence > 0.5:
            print("Pedestrian detected! Slow down.")
            speak("Pedestrian detected! Slow down.")

        elif label == 'car' and confidence > 0.5:
            print("Vehicle detected ahead. Maintain a safe distance.")
            speak("Vehicle detected ahead. Maintain a safe distance.")

        elif label == 'traffic light' and confidence > 0.5:
            print("Traffic light detected!")
            speak("Traffic light detected!")

        # You can add more conditions based on other objects like cyclists, road signs, etc.
        if label == 'stop sign' and confidence > 0.5:
            print("Stop sign detected! Prepare to stop.")
            speak("Stop sign detected! Prepare to stop.")

#Real-Time Processing: Continuously capture frames from the video feed, detect objects in real-time, make decisions based on the detected objects, and give audio feedback.

def process_frame():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Detect objects in the current frame
        detections = detect_objects(frame)

        # Apply decision-making logic based on detections
        decision_making(detections)

        # Display the frame with detection results (bounding boxes)
        frame_with_boxes = frame.copy()
        for index, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name']
            confidence = row['confidence']
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_with_boxes, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the frame with detected bounding boxes
        cv2.imshow("Object Detection", frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#Use Threads for Parallel collective Tasks: Since detecting objects, making decisions, and giving audio recommendations should happen concurrently, use Python's threading to run these tasks in parallel collective.

    def run_system():
        detection_thread = threading.Thread(target=process_frame)
        detection_thread.start()

    if __name__ == "__main__":
        run_system()

#Final Thoughts:

    #Real-time Performance: the used YOLO++ model's inference time is fast enough for real-time processing in our hardware i3.
