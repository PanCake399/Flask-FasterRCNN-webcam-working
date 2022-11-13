import torch
import torch.nn as nn
import cv2
import numpy as np
from models.create_fasterrcnn_model import create_model
import torchvision.transforms as transforms
from utils.transforms import infer_transforms
from utils.annotations import inference_annotations, annotate_fps
import time

from flask import Flask, Response, render_template, url_for

app = Flask(__name__)

checkpoint = torch.load('best_model.pth', map_location='cpu') # If config file is not given, load from model dictionary.
     
data_configs = True
NUM_CLASSES = checkpoint['config']['NC']
CLASSES = checkpoint['config']['CLASSES']

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

build_model = create_model['fasterrcnn_resnet50_fpn']
model = build_model(num_classes=NUM_CLASSES, coco_model=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to('cpu').eval()



def gen():
    detection_threshold = 0.5
    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.
    
    cap=cv2.VideoCapture(0)

    while(cap.isOpened()):

        success,frame = cap.read(cv2.CAP_DSHOW)
        frame = cv2.flip(frame, 1)
        if success:
            image = frame.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = infer_transforms(image)
            image = torch.unsqueeze(image, 0)

            start_time = time.time()
            with torch.no_grad():
                # Get predictions for the current frame.
                outputs = model(image.to('cpu'))
                print(f"outputs : {type(outputs)}")

            forward_end_time = time.time()
        
            forward_pass_time = forward_end_time - start_time
            # Get the current fps.
            fps = 1 / (forward_pass_time)
            # Add `fps` to `total_fps`.
            total_fps += fps
            # Increment frame count.
            frame_count += 1

            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

            if len(outputs[0]['boxes']) != 0:
                frame = inference_annotations(
                    outputs, detection_threshold, CLASSES,
                    COLORS, frame
                )
            frame = annotate_fps(frame, fps)

            final_end_time = time.time()
            forward_and_annot_time = final_end_time - start_time
            print_string = f"Frame: {frame_count}, Forward pass FPS: {fps:.3f}, "
            print_string += f"Forward pass time: {forward_pass_time:.3f} seconds, "
            print_string += f"Forward pass + annotation time: {forward_and_annot_time:.3f} seconds"
            print(print_string)  
            
        else:
            break
        
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')




@app.route('/')
@app.route('/index')
def index():
    
    return render_template('index.html')

@app.route("/video")
def video():
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False)