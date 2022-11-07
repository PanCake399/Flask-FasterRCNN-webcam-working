import torch
import torch.nn as nn
import cv2
import numpy as np
from models.create_fasterrcnn_model import create_model
import torchvision.transforms as transforms
from utils.transforms import infer_transforms
from utils.annotations import inference_annotations


checkpoint = torch.load('best_model.pth', map_location='cpu')
        # If config file is not given, load from model dictionary.
     
data_configs = True
NUM_CLASSES = checkpoint['config']['NC']
CLASSES = checkpoint['config']['CLASSES']

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

build_model = create_model['fasterrcnn_resnet50_fpn']
model = build_model(num_classes=NUM_CLASSES, coco_model=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to('cpu').eval()

detection_threshold = 0.5
cap = cv2.VideoCapture(0)
width = 800
height = 500

RESIZE_TO = (width, height)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, RESIZE_TO)
        image = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = infer_transforms(image)
        image = torch.unsqueeze(image, 0)
      
        with torch.no_grad():
                # Get predictions for the current frame.
                outputs = model(image.to('cpu'))
        # Load all detection to CPU for further operations.
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        
        if len(outputs[0]['boxes']) != 0:
                frame = inference_annotations(
                    outputs, detection_threshold, CLASSES,
                    COLORS, frame
                )
        cv2.imshow('Prediction', frame)
            # Press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()