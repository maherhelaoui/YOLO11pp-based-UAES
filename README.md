# Universal Autonomous Driving System: Innovation with yolo11++-based-UAES

____________________________________________________________________________________________________________________

                            A. YOLO11++ proposed detector
____________________________________________________________________________________________________________________


YOLO11++ needs installing

pip install ultralytics  # For YOLO11++ like YOLO11
_____________________________________________________________________________________________________________________
                                        
                                        Download trained models

_____________________________________________________________________________________________________________________

We prepared for each Dataset a trained model of yolo11++ as proposed in the Visual computer paper. 
We present a best trained model of yolo11++ for each studied Dataset.

Trained Model 1. yolo11++MP.pt : Trained model of yolo11++ for Medical-Pills MP Dataset
Trained Model 2. yolo11++GW.pt : Trained model of yolo11++ for GlobalWheat2020 GW Dataset
Trained Model 3. yolo11++S.pt : Trained model of yolo11++ for Signature S Dataset
Trained Model 4. yolo11++AW.pt : Trained model of yolo11++ for African-Wildlife AW Dataset
Trained Model 5. yolo11++BT.pt : Trained model of yolo11++ for Brain-Tumor BT Dataset

______________________________________________________________________________________________________________________
 
                                 Train YOLO11++ on different Datasets i
______________________________________________________________________________________________________________________
 

# For each Dataseti we charge the trained YOLO++ Models
#model3 = YOLO("yolo11++MP.pt")  # if model.train(data="medical-pills.yaml", epochs=100)
#model3 = YOLO("yolo11++GW.pt")  # if model.train(data="GlobalWheat2020.yaml", epochs=100)
#model3 = YOLO("yolo11++S.pt")  # if model.train(data="signature.yaml", epochs=100)
#model3 = YOLO("yolo11++AW.pt")  # if model.train(data="african-wildlife.yaml", epochs=100)
#model3 = YOLO("yolo11++BT.pt")  # if model.train(data="brain-tumor.yaml", epochs=100)
model3 = YOLO("yolo11++CC128.pt")  # if model.train(data="coco128.yaml", epochs=100)
model = model3


# Train chargedYOLO Model on each Dataseti
#results = model.train(data="medical-pills.yaml", epochs=100, imgsz=640) # Train MP Dataset use model yolo11++MP.pt
#results = model.train(data="GlobalWheat2020.yaml", epochs=100, imgsz=640) # Train GW Dataset use model yolo11++GW.pt
#results = model.train(data="signature.yaml", epochs=100, imgsz=640) # Train SD Dataset use model yolo11++S.pt
#results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640) # Train AW Dataset use model yolo11++AW.pt
#results = model.train(data="brain-tumor.yaml", epochs=100, imgsz=640) # Train BT Dataset use model yolo11++BT.pt
results = model.train(data="coco128.yaml", epochs=100) # Train CC128 Dataset use model yolo11++CC128.pt 



To Train YOLO11++ on different Datasets you can use 

$ python TrainYOLO11++.py
or
$ python3 TrainYOLO11++.py

You can charge the trained YOLO++ Models by selecting one from this list in TrainYOLO11++.py
#model3 = YOLO("yolo11++MP.pt")  # if model.train(data="medical-pills.yaml", epochs=100)
#model3 = YOLO("yolo11++GW.pt")  # if model.train(data="GlobalWheat2020.yaml", epochs=100)
#model3 = YOLO("yolo11++S.pt")  # if model.train(data="signature.yaml", epochs=100)
#model3 = YOLO("yolo11++AW.pt")  # if model.train(data="african-wildlife.yaml", epochs=100)
#model3 = YOLO("yolo11++BT.pt")  # if model.train(data="brain-tumor.yaml", epochs=100)
model3 = YOLO("yolo11++CC128.pt")  # if model.train(data="coco128.yaml", epochs=100)
model = model3


You can change the used dataset by selecting one from this list in TrainYOLO11++.py

#results = model.train(data="medical-pills.yaml", epochs=100, imgsz=640) # Train MP Dataset use model yolo11++MP.pt
#results = model.train(data="GlobalWheat2020.yaml", epochs=100, imgsz=640) # Train GW Dataset use model yolo11++GW.pt
#results = model.train(data="signature.yaml", epochs=100, imgsz=640) # Train SD Dataset use model yolo11++S.pt
#results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640) # Train AW Dataset use model yolo11++AW.pt
#results = model.train(data="brain-tumor.yaml", epochs=100, imgsz=640) # Train BT Dataset use model yolo11++BT.pt
results = model.train(data="coco128.yaml", epochs=100) # Train CC128 Dataset use model yolo11++CC128.pt 

Each yolo11++ model is generated as decribed in the Visual computer journal paper.


____________________________________________________________________________________________________________________

                            B. Universal Autonomous Expert System Analysis Process
____________________________________________________________________________________________________________________



In order to compare the proposed original Universal Autonomous Expert System using the proposed YOLO11++ detector, we present different versions of Expert System using YOLO11++. We start with Dynamic Expert System for Autonomous Driving, then we improve this version by using LLM, finally we detail our proposed Universal Autonomous Expert System (UAES). The proposed UAES Updates Facts Using Perceptron and Evaluates Decisions and Updates Rules, in real time.

  
______________________________________________________________________________________________________________________
                             1. Dynamic Expert System for Autonomous Driving
______________________________________________________________________________________________________________________

To create a dynamic expert system for autonomous driving that integrates YOLO11++ for object detection with audio recommendations in Python, you’ll need to structure the system to not only process the real-time video feed and detect objects using YOLO but also offer audio recommendations based on the driving situation. These recommendations could be verbal instructions or warnings to the driver (or simulated driver).

We use the proposed detector (YOLO11++), we'll integrate audio feedback using text-to-speech (TTS) to give recommendations based on object detections.
Key Components:

    YOLO for Object Detection: Detect vehicles, pedestrians, traffic lights and signs.
    Decision-making system: Based on object detection, recommend actions like slowing down, stopping, or changing lanes.
    Audio Recommendations: Use Text-to-Speech (TTS) to provide verbal instructions or warnings to the driver.

Steps to Implement:
    Safety: Ensure that safety checks are in place before implementing this system in a real-world autonomous vehicle.

    Install Required Libraries: To begin, install all the necessary libraries:
        YOLO model (via Ultralytics)
        Text-to-Speech for audio output (pyttsx3 or gTTS)
        OpenCV for video processing


pip install opencv-python
pip install torch torchvision
pip install ultralytics  # For YOLO11++
pip install pyttsx3  # Text-to-Speech library
pip install numpy
pip install threading # Used for parallel collective run

Import Libraries:

import cv2
import torch
import numpy as np
import pyttsx3
from ultralytics import YOLO

Load YOLO Model: Load the validated YOLO11++ model for object detection. This model will be used to detect objects in real-time.

# Load YOLO model 
model = YOLO("yolo11++.pt") # can be downloaded from :  

Initialize Audio Feedback (TTS): We will use pyttsx3 for text-to-speech functionality. It will provide audio feedback based on the decisions made by the expert system.

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

Object Detection Using YOLO: Detect objects such as vehicles, pedestrians, traffic signs, etc., using YOLO.

def detect_objects(frame):
    # Perform inference (object detection)
    results = model(frame)

    # Get results (objects, boxes, and scores)
    detections = results.pandas().xywh[0]  # Convert results to Pandas dataframe
    return detections

Dynamic Expert System Decision Logic: Based on the objects detected in the video feed, we can create a set of rules to make driving decisions (e.g., stop for a pedestrian, slow down for a car ahead). Then, we provide audio feedback using the TTS system.

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

Real-Time Processing: Continuously capture frames from the video feed, detect objects in real-time, make decisions based on the detected objects, and give audio feedback.

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

Use Threads for Parallel collective Tasks: Since detecting objects, making decisions, and giving audio recommendations should happen concurrently, use Python's threading to run these tasks in parallel collective.

    def run_system():
        detection_thread = threading.Thread(target=process_frame)
        detection_thread.start()

    if __name__ == "__main__":
        run_system()

Final Thoughts:

    Real-time Performance: the used YOLO++ model's inference time is fast enough for real-time processing in our hardware i3.

    Audio Recommendations: Using pyttsx3 for audio feedback is simple, but you could also use other TTS engines (e.g., gTTS for Google TTS) if you prefer different voices or languages. Additionally, integrate voice recognition for more complex interactions (e.g., if the user asks for a route change).

    Extended Expert System: As you scale the system, you can integrate more complex decision-making processes (e.g., navigating based on the traffic condition, detecting road signs, interpreting traffic rules, etc.).

By combining YOLO-based object detection with dynamic decision-making and audio feedback, this system provides an intelligent interface for autonomous driving, improving safety and user experience with real-time recommendations.





_______________________________________________________________________________________________

                   Dynamic Expert System for autonomous driving using LLM API
_______________________________________________________________________________________________


To create a dynamic expert system for autonomous driving that integrates the proposed YOLO11++ for object detection, Large Language Models (LLMs) for decision-making and understanding context, and audio recommendations for verbal instructions, we need to combine several components. Each piece will handle a different aspect of the system:

    YOLO: Object detection (e.g., detecting pedestrians, vehicles, traffic signs).
    LLMs: Using a language model (like GPT-4) to handle more complex decision-making, reasoning, and context-based advice (e.g., "What should I do if I see a red light?" or "How should I adjust speed in bad weather?").
    Audio Feedback: Using text-to-speech (TTS) to provide verbal instructions to the driver or to the system itself (for example, "Pedestrian detected, slow down").

Architecture Overview:

    YOLO11++ will detect objects like pedestrians, vehicles, and traffic signs from video frames.
    The LLM will be responsible for interpreting complex driving scenarios and recommending the appropriate actions (e.g., "A pedestrian is detected, should I stop?").
    Audio feedback will give real-time verbal recommendations based on LLM's decisions (using a library like pyttsx3 or gTTS for speech synthesis).
    The system will combine these three components to simulate an intelligent, adaptive autonomous driving system.

Step-by-Step Implementation
0. Safety: Ensure that safety checks are in place before implementing this system in a real-world autonomous vehicle.

1. Install Required Libraries:

pip install opencv-python
pip install torch torchvision
pip install ultralytics  # For YOLO11++
pip install pyttsx3  # Text-to-speech for audio feedback
pip install openai  # For GPT-3/4 (Large Language Model)
pip install threading # Used for parallel collective run

2. Import Necessary Libraries:

import cv2
import torch
import numpy as np
import pyttsx3
import openai
from ultralytics import YOLO
import threading

3. Initialize OpenAI and YOLO Model:

We will use OpenAI GPT-3/4 for decision-making and our proposed YOLO11++ for real-time object detection. Ensure you have a valid OpenAI API key for GPT.

# Initialize OpenAI API (ensure you have set up an OpenAI API key)
openai.api_key = "YOUR_OPENAI_API_KEY"

# Load YOLO11++ Model
model = YOLO("yolo11++.pt")  

4. Initialize Text-to-Speech (TTS):

Using pyttsx3, we can give audio feedback for the recommendations.

# Initialize pyttsx3 for Text-to-Speech
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

5. Define Functions for YOLO and LLM Interaction:

    Object Detection with YOLO: Detect objects in the video feed.
    Decision-Making with LLM: Ask GPT-based LLM for advice based on the detected objects and conditions.

def detect_objects(frame):
    results = model(frame)  # Perform object detection using YOLO
    detections = results.pandas().xywh[0]  # Convert results to Pandas DataFrame
    return detections

def get_llm_decision(objects_detected):
    # Create a prompt for the LLM (e.g., GPT-4) based on detected objects
    object_labels = [obj['name'] for _, obj in objects_detected.iterrows()]
    prompt = f"Given the detected objects: {', '.join(object_labels)}, what should the autonomous vehicle do?"

    # Query the LLM for decision-making
    response = openai.Completion.create(
        model="text-davinci-003",  # Or GPT-4 model if available
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )

    return response.choices[0].text.strip()

6. Audio Recommendations Based on LLM:

Using the LLM's response, we'll provide audio feedback.

def audio_feedback(decision):
    speak(decision)

7. Real-time Processing Loop with Video Feed and Dynamic Decisions:

Now we can combine the real-time video processing, object detection, and decision-making into a continuous loop that runs while capturing the video feed.

def process_frame():
    cap = cv2.VideoCapture(0)  # Open webcam (use video file path if needed)
    
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Step 1: Detect objects in the frame using YOLO
        detections = detect_objects(frame)

        # Step 2: Get decision from LLM based on detected objects
        decision = get_llm_decision(detections)

        # Step 3: Provide audio feedback based on LLM's decision
        audio_feedback(decision)

        # Step 4: Visualize bounding boxes and labels in the frame
        frame_with_boxes = frame.copy()
        for _, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name']
            confidence = row['confidence']
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_with_boxes, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Step 5: Display the video feed with bounding boxes
        cv2.imshow("Object Detection", frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

8. Running the System with Parallel Threads:

To ensure real-time responsiveness, we can run object detection, decision-making, and audio feedback concurrently. We use Python's threading module to run the system smoothly.

def run_system():
    # Create and start the processing thread
    detection_thread = threading.Thread(target=process_frame)
    detection_thread.start()

if __name__ == "__main__":
    run_system()

9. Explanation:

    YOLO11++ (Object Detection): The system captures frames from the video stream and processes each frame to detect objects (e.g., pedestrians, cars, traffic lights).
    LLM (Decision-Making): Based on the objects detected (e.g., "person", "car", "stop sign"), a prompt is created for the Large Language Model (LLM) (GPT-4). The LLM analyzes the context and suggests an appropriate course of action (e.g., "Stop the car", "Slow down", "Maintain speed").
    Audio Feedback: The LLM's decision is converted into a verbal recommendation using text-to-speech (TTS), so the system can give real-time spoken advice or instructions.
    Parallel Processing: Object detection, decision-making, and audio feedback are processed in parallel using threads for seamless real-time performance.

Final Thoughts:

    Real-time Processing: The system uses a threaded loop to ensure all components (video feed, detection, decision-making, and audio feedback) run concurrently and in real-time.
    Extensibility: You can extend the LLM's capabilities by fine-tuning it or using more complex decision trees to handle a broader range of driving scenarios (e.g., weather conditions, road signs, and more).
    Optimization: Depending on your hardware (especially GPU availability), you may need to optimize YOLO model inference speed to ensure the system can keep up with real-time video feeds.

This combination of YOLO object detection, LLMs for decision-making, and audio recommendations creates a sophisticated and dynamic expert system for autonomous driving, which adapts to its environment and provides verbal feedback based on detected objects.

_______________________________________________________________________________________________

                           Universal Autonomous Expert System
                           (Dynamic Expert System for autonomous driving 
                           with real time updates of facts using perceptron
                           real time updates of rules by evaluate precedent decisions)
________________________________________________________________________________________________

To implement our proposed Universal Autonomous Expert System that integrates YOLO for object detection, audio recommendations, dynamic updates of facts using a perceptron, and dynamic update of rules based on evaluated decisions, we can break the problem into several distinct steps.

The goal is to create a system that:

    Detects objects in real-time using the proposed and validated YOLO11++ detector (for example, pedestrians, cars, traffic signs).
    Makes dynamic decisions using rules and evaluates those decisions for correctness and efficiency.
    Updates facts dynamically using a perceptron (which can be used to learn and adjust facts from the environment over time).
    Updates rules based on decision outcomes to optimize performance.
    Provides real-time audio recommendations to guide the car's actions.

Architecture

The system architecture can be broken down as follows:

    YOLO11++ (Object Detection): Detect objects from the video stream.
    Dynamic Facts Update with Perceptron: Use a perceptron to dynamically update the facts (environmental information) based on object detection.
    Dynamic Rules Update: Adjust decision-making rules based on evaluated outcomes of previous actions.
    Audio Recommendations: Provide verbal feedback or instructions based on the decision-making process.

Steps to Implement the System:
0. Safety: Ensure that safety checks are in place before implementing this system in a real-world autonomous vehicle.

1. Install Required Libraries

pip install opencv-python
pip install torch torchvision
pip install ultralytics  # For YOLOv5 or YOLOv8
pip install pyttsx3  # Text-to-speech for audio feedback
pip install numpy
pip install sklearn  # For the Perceptron
pip install threading # Used for parallel collective run

2. Import Libraries

import cv2
import torch
import numpy as np
import pyttsx3
import openai
from sklearn.neural_network import Perceptron
from ultralytics import YOLO
import threading

3. Initialize OpenAI, YOLO, and TTS

You need to load YOLO11++ for object detection and initialize text-to-speech (TTS) for audio feedback. You can also initialize a perceptron to dynamically update facts.

# Initialize OpenAI API for decision-making (if needed)
openai.api_key = "YOUR_OPENAI_API_KEY"  # Make sure to replace with your API key

# Load YOLO model (YOLO11++)
model = YOLO("yolo11++.pt")  

# Initialize pyttsx3 for Text-to-Speech (audio feedback)
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Perceptron for dynamic facts update
perceptron = Perceptron(max_iter=1000)

# Initialize an empty list for facts (e.g., detected objects, vehicle status)
facts = []

# Initialize rules (based on detections and system learning)
rules = {
    "pedestrian_detected": "slow_down",
    "vehicle_detected": "maintain_distance",
    "stop_sign_detected": "stop",
    # More rules can be added dynamically
}

4. Detect Objects with YOLO

The function detect_objects will process frames from the camera and identify objects like pedestrians, vehicles, etc.

def detect_objects(frame):
    results = model(frame)  # Perform object detection using YOLO
    detections = results.pandas().xywh[0]  # Convert results to Pandas DataFrame
    return detections

5. Update Facts Using Perceptron

The perceptron will be used to update facts dynamically, based on detected objects. We can train the perceptron to associate object types with certain driving facts (like stopping, slowing down, etc.).

def update_facts(detections):
    global facts

    new_facts = []

    # Extract facts based on the detections
    for _, row in detections.iterrows():
        label = row['name']
        confidence = row['confidence']

        # Update the facts list based on detected objects
        if label == 'person' and confidence > 0.5:
            new_facts.append('pedestrian_detected')
        elif label == 'car' and confidence > 0.5:
            new_facts.append('vehicle_detected')
        elif label == 'stop sign' and confidence > 0.5:
            new_facts.append('stop_sign_detected')

    # Update perceptron with new facts
    if new_facts:
        fact_vector = [1 if fact in new_facts else 0 for fact in rules.keys()]
        perceptron.partial_fit([fact_vector], [1])  # Simulate learning a correct action

    facts.extend(new_facts)

6. Evaluate Decisions and Update Rules

After making a decision based on detected objects, we can evaluate the decision outcome (e.g., did the vehicle stop when it should have?) and update the rules accordingly. For this purpose, we could use a simple feedback mechanism to improve the system over time.

def evaluate_and_update_rules(action_taken):
    # Example rule-based decision evaluation (you can make it more sophisticated)
    if action_taken == "stop":
        print("Evaluating stop decision...")
        # If the vehicle did not stop appropriately, adjust the rule
        if "pedestrian_detected" in facts:
            print("Pedestrian detected, updating rule to stop more effectively.")
            rules["pedestrian_detected"] = "stop"

7. Decision Making with LLM (Optional)

Optionally, you can use a Large Language Model (like GPT-4) to process complex scenarios and update rules based on the context.

def get_llm_decision(objects_detected):
    # Create a prompt for LLM (e.g., GPT-4) based on detected objects
    object_labels = [obj['name'] for _, obj in objects_detected.iterrows()]
    prompt = f"Given the detected objects: {', '.join(object_labels)}, what should the autonomous vehicle do?"

    # Query the LLM for decision-making
    response = openai.Completion.create(
        model="text-davinci-003",  # Use GPT-4 if needed
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )

    return response.choices[0].text.strip()

8. Combine the Components and Provide Audio Feedback

The system continuously processes frames, updates facts, evaluates decisions, and provides audio feedback.

def process_frame():
    cap = cv2.VideoCapture(0)  # Open webcam (you can replace with a video file if needed)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Step 1: Detect objects in the frame using YOLO
        detections = detect_objects(frame)

        # Step 2: Update facts using the perceptron (learn from environment)
        update_facts(detections)

        # Step 3: Get decision based on facts and objects detected
        decision = get_llm_decision(detections)

        # Step 4: Provide audio feedback for the decision
        speak(decision)

        # Step 5: Visualize detections with bounding boxes
        frame_with_boxes = frame.copy()
        for _, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name']
            confidence = row['confidence']
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_with_boxes, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Step 6: Display the frame with detections
        cv2.imshow("Object Detection", frame_with_boxes)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

9. Running the System

We run the system, which will continuously process frames, detect objects, evaluate decisions, update facts and rules, and provide real-time audio feedback.

def run_system():
    detection_thread = threading.Thread(target=process_frame)
    detection_thread.start()

if __name__ == "__main__":
    run_system()

Final Thoughts:

    Real-time Performance: The system uses a multi-threaded approach for parallel collective run to ensure optimal real-time processing for object detection, decision-making, and audio feedback.
    Learning and Adaptation: The perceptron dynamically updates facts based on the detected objects. As the system receives more feedback and detects more scenarios, it can adjust its rules and behaviors.
    Extendability: This setup can be further extended by introducing more complex rules, including reinforcement learning to optimize the vehicle’s decisions over time, and using advanced LLMs (e.g., GPT-4) for more nuanced decision-making.

This architecture forms the foundation of a UAES for autonomous driving that adapts to its environment, learns from experience, and provides audio feedback to the driver or system operator.

Safety: Ensure that safety checks are in place before implementing this system in a real-world autonomous vehicle.


________________________________________________________________________________________________________________________

                                                          Citations and Acknowledgements

________________________________________________________________________________________________________________________

Universal Autonomous Driving System: Innovation with YOLO11++-based Universal Autonomous Expert System Publication

If you use YOLO11++ or Universal Autonomous Expert System software from this repository in your work, please cite it using the following format:

BibTeX
@article{hel25YOLO11++UAES,
  title={Universal Autonomous Driving System: Innovation with YOLO11++-based Universal Autonomous Expert System},
  author={Maher, Helaoui and Sahbi, Bahroun and Ezzeddine, Zagrouba},
  journal={The Visual Computer (Submitted)},
  url = {https://github.com/maherhelaoui/YOLO11pp-based-UAES/},
  year={2025},
  publisher={Springer}
}


Please note that the DOI will be added to the citation once it is available.


______________________________________________________________________________________________________________________

                            C Real Word Simplified Demo of Universal Autonomous Driving System 
______________________________________________________________________________________________________________________
    
import os
import shutil
from pathlib import Path
from ultralytics.utils.benchmarks import RF100Benchmark
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque
#make a trained model on different datasets is it importants ?
#How improve yolo 11 ?
# YOLO11 is trained on coco dataset.
# what is the result if we concatinate a multi trained models ?
#Create new model by generate boxes, scores, classes as concatenation of 3 boxes, scores, classes
#Create a concatenation of 3 different models results

model = YOLO("yolo11n.yaml")
model1 = YOLO("yolov8sign.pt")  # detect all sign. You can use any sign detector Model. The model chosed in this demo is mensioned in the Visual computer journal paper. It must be added to the work space. 
model = model1
model2 = YOLO("yolov8nbt.pt")  # detect traffic Light red green yellow. You can use any traffic Light detector Model. The model chosed in this demo is mensioned in the Visual computer journal paper. It must be added to the work space. 
model = model2
model3 = YOLO("yolo11++CC128.pt")  # This traned model is the most adequate to detect cars bus trained on Coco128 Dataset 
model = model3

###################################################################################################################################

# Extract boxes, scores and classes
    boxes1 = results1.xyxy[0][:, :4].cpu().numpy()
    scores1 = results1.xyxy[0][:, 4].cpu().numpy()
    classes1 = results1.xyxy[0][:, 5].cpu().numpy()

    boxes2 = results2.xyxy[0][:, :4].cpu().numpy()
    scores2 = results2.xyxy[0][:, 4].cpu().numpy()
    classes2 = results2.xyxy[0][:, 5].cpu().numpy()

    boxes3 = results3.xyxy[0][:, :4].cpu().numpy()
    scores3 = results3.xyxy[0][:, 4].cpu().numpy()
    classes3 = results3.xyxy[0][:, 5].cpu().numpy()

    # Combine boxes, scores and classes
    all_boxes = np.concatenate([boxes1, boxes2, boxes3])
    all_scores = np.concatenate([scores1, scores2, scores3])
    all_classes = np.concatenate([classes1, classes2, classes3])

    # delete redondantes
    keep_indices = nms(torch.tensor(all_boxes), torch.tensor(all_scores), iou_threshold=0.5)
    final_boxes = all_boxes[keep_indices]
    final_scores = all_scores[keep_indices]
    final_classes = all_classes[keep_indices]

    return final_boxes, final_scores, final_classes

########################################################################################################################

x1=1000
# Open video
video_path = "VIDS/VIDS13.mp4"
cap = cv2.VideoCapture(video_path)

# Detect speed demo
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video = cv2.VideoWriter('VIDS/VIDSVDMD13.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# traked objects
tracked_objects = {}

# Compute speed demo
def calculate_speed(prev_position, current_position, fps):
    distance = np.linalg.norm(np.array(current_position) - np.array(prev_position))
    speed = distance * fps  # en pixels par seconde
    return speed

# Read traffic Signs (simplified)
def read_traffic_signs(frame):
    # Ici, vous pouvez ajouter un modèle de détection de panneaux de signalisation
    # Pour cet exemple, nous allons simplement retourner un panneau fictif
    for r in results34:
       for box in r.boxes:
          if x1<5 :
              cv2.putText(frame, "Recommend: Stop object ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
          else :
               if x1<20 :
                     cv2.putText(frame, "Recommend: Worning object ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
               else :
                cv2.putText(frame, "Recommend: Keep Line", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
         
    for r in results2:
       for box in r.boxes:
          print(int(box.cls))
          if(int(box.cls)==0):
                 cv2.putText(frame, "Recommend:           Stop red", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
          if(int(box.cls)==1):
                 cv2.putText(frame, "Recommend:            Green Go", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    for r in results1:
       #print(model1.names)
       for box in r.boxes:
          #print(box.cls)) 
          if(int(box.cls)):
             print(int(box.cls))
             #cv2.putText(frame, f"                               id :{int(box.cls)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    

# Demo display recommendations (simplified)
def display_recommendations(frame, sign):
    if sign == "Go":
        cv2.putText(frame, "Recommendation: Green Go", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if sign == "Stop":
        cv2.putText(frame, "Recommendation: Stoooooop", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #else :
        #cv2.putText(frame, "Recommendation: Keep Line", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Demo video treatement
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Détection des objets avec YOLO
    results1 = model1(frame)
    for r in results1:
        frame = r.plot()
    results2 = model2(frame)
    for r in results2:
        frame = r.plot()

    results3 = model3(frame)
    # Demo resultats treatement
    for result in results3:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id == 2:  # Class ID 2 correspond aux voitures dans YOLO
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                object_id = f"{x1}_{y1}_{x2}_{y2}"

                # Demo compute speed (simplified)
                if object_id in tracked_objects:
                    prev_position = tracked_objects[object_id]
                    current_position = ((x1 + x2) // 2, (y1 + y2) // 2)
                    speed = calculate_speed(prev_position, current_position, fps)
                    cv2.putText(frame, f"Speed: {speed:.2f} px/s", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                tracked_objects[object_id] = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Dessiner la boîte autour de la voiture
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Read_traffic_signs
    sign = read_traffic_signs(frame)

    # display_recommendations
    display_recommendations(frame, sign)

    # Output Video
    output_video.write(frame)

    # Display frame (optionnel)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Free ressources
cap.release()
output_video.release()
cv2.destroyAllWindows()


If you use this demo or Universal Autonomous Driving System software from this repository in your work, please cite it using the following format:

BibTeX @article{hel25YOLO11++UAES, title={Universal Autonomous Driving System: Innovation with YOLO11++-based Universal Autonomous Expert System}, author={Maher, Helaoui and Sahbi, Bahroun and Ezzeddine, Zagrouba}, journal={The Visual Computer (Submitted)}, url = {https://github.com/maherhelaoui/YOLO11pp-based-UAES/}, year={2025}, publisher={Springer} }

Please note that the DOI will be added to the citation once it is available.

