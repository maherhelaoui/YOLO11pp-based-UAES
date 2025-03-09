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
model1 = YOLO("yolov8sign.pt")  # detect all sign 
model = model1
model2 = YOLO("yolov8nbt.pt")  # detect traffic Light red green yellow
model = model2
model3 = YOLO("yolo11++CC128.pt")  # detect cars bus trained on Coco Dataset 319 layers, 2624080 parameters, 2624064 gradients
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

###################################################################################################################################

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








