"""
    Agent tracking: detects the agents in a video stream (from a camera or a video file)
    and splits it into smaller streams which track each individual agent

    Agents are detected using the  darknet implementation of YOLOv4
    (https://github.com/AlexeyAB/darknet)
"""


import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import darknet

from math import sqrt
import numpy as np
import cv2
import sys
import itertools
import time

from tracking_utils import *


# Initialize model

class Net(torch.nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        #self.backbone = models.densenet161()
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.classifiers = [torch.nn.Linear(1000,1) for _ in range(4)]
        self.classifiers = torch.nn.ModuleList(self.classifiers)
        
    def forward(self,x):
        x = self.backbone(x)
        sigmoid = torch.nn.Sigmoid()
        x1,x2,x3,x4 = [sigmoid(f(x)) for f in self.classifiers]
        x = torch.cat((x1,x2,x3,x4),dim=1)
        return x

DIMS = [3,224,224]

# Normalization params for torchvision models
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Normalization layer
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True
print("USE_CUDA:",use_cuda)

model = Net()
model.load_state_dict(torch.load("model_weights"))
model = model.cuda()

# Initialize tracking
identities = []
prev_assignment = dict()
expected_positions = dict()
tolerance = dict()
delta = dict()
A = 0
tol = 50

# Initialize YOLO
net, class_names, colors = darknet.load_network('cfg/yolov4.cfg', 'cfg/coco.data', 'yolov4.weights')


# Net height and width
nh = darknet.network_height(net)
nw = darknet.network_width(net)
img_buffer = darknet.make_image(nw, nh, 3)

# Start webcam
cap = cv2.VideoCapture(0)

# Original (camera) width and height
ow  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
oh = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float


print("Starting capture. Press Q to stop.")
ite = 0
t1 = time.time()

while(True):
    # Capture frame-by-frame

    ret, frame = cap.read()

    frame_resized = cv2.resize(frame,
                               (nw, nh),
                               interpolation=cv2.INTER_LINEAR)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    darknet.copy_image_from_bytes(img_buffer,frame_rgb.tobytes())
    detections = darknet.detect_image(net, class_names, img_buffer, thresh=0.5)

    # Only consider persons as agents
    
    agent_categories = ['person'] 
    detections = list(filter(lambda x: x[0] in agent_categories, detections))

    # New centers
    centers = [np.array((box[0],box[1],box[2],box[3])) for cat,conf,box in detections]
    
    # We will print the bboxes over this image
    image = frame
    
    subframe_idens = []
    subframes = []

    # Assign, print boxes and write to streams
    assignment_dict,assignment_list,A = naive_track(prev_assignment, expected_positions, centers, tolerance, tol=tol, A=A)        
    expected_positions = dict() # Clean every iteration (possible residue from previous assignment)
    for i in assignment_dict.keys():
        if i in prev_assignment:
            current,prev = assignment_dict[i],prev_assignment[i]
            expected_positions[i] = current+(current-prev) # unnecessary to use extra memory
        else:
            expected_positions[i] = assignment_dict[i]
    
    prev_assignment = assignment_dict
    
    for iden,bbox in assignment_dict.items():
        x = int(bbox[0] - bbox[2]/2)
        y = int(bbox[1] - bbox[3]/2)
        w = int(bbox[2])
        h = int(bbox[3])
        # Get bounding box in original coordinates
        x,y,w,h = transform_bbox(x,y,w,h,nw,nh,ow,oh)
        x = max(0,x)
        y = max(0,y)
        w=max(1,w)
        h=max(1,h)
        # Write subframe to agent stream
        subframe = frame[y:y+h,x:x+w]
        
        subframes.append(preprocess(Image.fromarray(subframe, 'RGB')))
    
    # Inference
    input_frames = torch.stack(subframes)

    with torch.set_grad_enabled(False):
        res = model(input_frames.cuda()).cpu().numpy()

    print(res)
    for iden, irrs in zip(subframe_idens, res):
        print(iden)
        print(ires)
        print(np.average(ires))

    # Output
    cv2.imshow('frame',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    t2 = time.time()
    ite += 1
    if (t2-t1) > 10:
        print(ite/(t2-t1), "FPS")
        t1 = t2
        ite = 0

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
