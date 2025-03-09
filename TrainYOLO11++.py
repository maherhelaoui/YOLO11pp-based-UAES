import os
import shutil
from pathlib import Path
from ultralytics.utils.benchmarks import RF100Benchmark
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque


# YOLO11 is trained on coco dataset.
#Train YOLO11++ on different Datasets 

model = YOLO("yolo11n.yaml")
# Charger les modèles YOLO
model3 = YOLO("yolo11++.pt")  # detect cars bus trained on Coco Dataset 319 layers, 2624080 parameters, 2624064 gradients
model = model3



results = model.train(data="medical-pills.yaml", epochs=100, imgsz=640) # Train MP Dataset
#results = model.train(data="GlobalWheat2020.yaml", epochs=100, imgsz=640) # Train GW Dataset
#results = model.train(data="signature.yaml", epochs=100, imgsz=640) # Train SD Dataset
#results = model.train(data="african-wildlife.yaml", epochs=100, imgsz=640) # Train AW Dataset
#results = model.train(data="brain-tumor.yaml", epochs=100, imgsz=640) # Train BT Dataset
#results = model.train(data="coco128.yaml", epochs=100) # Train CC128 Dataset






