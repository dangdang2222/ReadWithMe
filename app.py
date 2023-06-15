from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from flask_cors import CORS, cross_origin
import base64
import io
import argparse
import numpy as np
import cv2
import time

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image, ImageTk
from utils import select_device, draw_gaze, gazeto3d
from PIL import Image, ImageOps

from face_detection import RetinaFace
from model import L2CS
import tkinter as tk

import matplotlib
from matplotlib import pyplot, image
import queue
q = queue.Queue()
app = Flask(__name__)
CORS(app)
frame_num = 0

init_x = []
init_y = []



@app.route('/process_image', methods=['POST'])
def process_image():
    global frame_num
    global init_x
    global init_y
    # Extract image data from request
    image_data = request.form.get('image')
    image_data = image_data.split(',', 1)[1]
    image_data = base64.b64decode(image_data)

    # Load the image with PIL
    image = Image.open(io.BytesIO(image_data))
    
    na = np.array(image)

    ###################################################################################
    with torch.no_grad():
        coord_list = []
        sigma = 50  # 가우시안 Blob의 표준 편차
        blob_intensity = 1.0  # Blob의 강도
        x_indices = np.arange(1600)
        y_indices = np.arange(1200)
        X, Y = np.meshgrid(x_indices, y_indices)
        frame = cv2.cvtColor(na, cv2.COLOR_RGB2BGR)
        frame=cv2.flip(frame,1)
        M = np.ndarray((1200, 1600))
        demo_img = np.ones((1600,1200))
        coordinates = []
        if frame is not None:
            faces = detector(frame)
            if faces is not None: 
                for box, landmarks, score in faces:
                    if score < .95:
                        continue
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])

                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min

                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img=transformations(im_pil)
                    img  = Variable(img).cuda(gpu)
                    img  = img.unsqueeze(0) 
                    
                    # gaze prediction
                    gaze_pitch, gaze_yaw = model(img)
                    
                    
                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)
                    
                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180
                    
                    pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                    yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0


                    p = pitch_predicted
                    y = yaw_predicted
                    r = 0

                    origin = np.array([0,0,40])
                    cy = np.cos(y)
                    sy = np.sin(y)
                    cr = np.cos(r)
                    sr = np.sin(r)
                    cp = np.cos(p)
                    sp = np.sin(p)

                    R_x = np.array([[1, 0, 0],
                                        [0, cp, -sp],
                                        [0, sp, cp]])

                    R_y = np.array([[cy, 0, sy],
                                        [0, 1, 0],
                                        [-sy, 0, cy]])

                    R_z = np.array([[cr, -sr, 0],
                                        [sr, cr, 0],
                                        [0, 0, 1]])

                    # 회전 행렬들의 곱으로 최종 회전 행렬 계산
                    rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
                    moved_point = np.dot(rotation_matrix, origin)
                    factor = (40/2.54*138)/ moved_point[2]
                    new_point = np.array([moved_point[0]*factor, moved_point[1]*factor,moved_point[2]*factor])

                    new_x = int(new_point[1]) + int(x_min + bbox_width/2.0)*1600/640
                    new_y = int(-1*new_point[0]) + int(y_min+bbox_height/3.0)*1200/480
                    new_z = int(new_point[2])
                    #coord_list.append([new_x, new_y])
                    if (frame_num<=30):
                        # cv2.putText(demo_img, 'See webcam', (x_min, y_max),cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0),2, cv2.LINE_AA)
                        # cv2.circle(demo_img, (int((x_min + bbox_width/2.0)/640 * 1600),int((y_min+bbox_height/3.0)/ 480 * 1200)),15,(255, 0,0),-1) #확인해야함
                        #cv2.rectangle(demo_img, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
                        if(frame_num>=10):
                            init_x.append(int(new_point[1])-int(1600/2)+int((x_min + bbox_width/2.0)/640 * 1600))
                            init_y.append(int(-1*new_point[0])+int((y_min+bbox_height/3.0)/ 480 * 1200))   
                        #coordinates.append([new_x, new_y]) ##점 좌표 예측
                    

                    elif(frame_num>30):
                        new_x = new_x - int(np.mean(init_x))
                        new_y = new_y -int(np.mean(init_y))  
                        blob_center = (new_x, new_y)  # Blob의 중심 좌표
                        gaussian_blob = blob_intensity * np.exp(-((X - blob_center[0]) ** 2 + (Y - blob_center[1]) ** 2) / (2 * sigma ** 2))
                        q.put(gaussian_blob)
                        M += gaussian_blob
                        if q.qsize() >=10:
                            M -= q.get()
                        sorted_indices = np.argsort(M.flatten())[::-1]
                        top_indices = sorted_indices[:10]
                        for index in top_indices:
                            x = index % M.shape[1]
                            y = index // M.shape[1]
                            coordinates.append([int(x),int(y)]) #가우시안 10개
    frame_num+=1
    return jsonify({'coordinates': coordinates})
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--img',dest='img_id', help='Got image input',  
        default='input/0.jpg', type=str)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

if __name__ == '__main__':

    cudnn.enabled = True
    arch='ResNet50'
    batch_size = 1
    gpu = select_device('0', batch_size=batch_size)
    snapshot_path = 'models/L2CSNet_gaze360.pkl'
   
    root = tk.Tk()
    root.withdraw()

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    model=getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()

    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x=0






    app.run(host='0.0.0.0', port=5000)
    #########################model loop######################3
