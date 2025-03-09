#If you use YOLO11++ model from this repository in your work, please cite it using the following format:

#BibTeX
#@article{hel25YOLO11++UAES,
#  title={Universal Autonomous Driving System: Innovation with YOLO11++-based Universal Autonomous Expert System},
#  author={Maher, Helaoui and Sahbi, Bahroun and Ezzeddine, Zagrouba},
#  journal={The Visual Computer (Submited)},
#  url = {[https://github.com/ultralytics/ultralytics](https://github.com/maherhelaoui/YOLO11pp-based-UAES/)},
#  year={2025},
#  publisher={Springer}
#}
#Please note that the DOI will be added to the citation once it is available. 



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






