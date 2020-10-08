import cv2
import numpy as np
import os
import time
import copy

from PIL import Image

import torch
import torchvision
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_model(config_path, weight_path):
    """
    Return YOLO model
    """
    net = cv2.dnn.readNetFromDarknet(config_path, weight_path)
    return net


def get_model_pytorch(weight_path):
    """
    Return pytorch model
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 30)

    # Load pre-trained model
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()
    return model


def detect_image_with_pytorch(img_name, img_path, weight_path, config_path):
    """
    Detect clothes of image
    :param img_name: file name
    :param img_path: directory path
    :param weight_path: weight path
    :param config_path: dont use
    :return: Detect image results, List of (cropped images path, cropped images classes)
    """
    label2class = {	0: 'coats',
                        1: 'rings',
                        2: 'handbag',
                        3: 'boots',
                        4: 'scarves',
                        5: 'socks and stockings',
                        6: 'shirts',
                        7: 'shirts hidden under jacket',
                        8: 'wallets purses',
                        9: 'coats',
                        10: 'jackets',
                        11: 'belts',
                        12: 'necklaces',
                        13: 'ties',
                        14: 'skirt',
                        15: 'sunglasses',
                        16: 'gloves and mitten',
                        17: 'backpacks',
                        18: 'monkeys',
                        19: 'pants',
                        20: 'shorts',
                        21: 'slopes',
                        22: 'bracelets',
                        23: 'clocks',
                        24: 'underwear',
                        25: 'hats',
                        26: 'swimsuits',
                        27: 'undefined',
                        28: 'dresses',
                        29: 'shoes'}
    num_classes = 30


    # Define the model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load pre-trained model
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()

    p = img_path+img_name

    img = Image.open(p).convert("RGB")
    img = T.ToTensor()(img)
    img = torch.unsqueeze(img, 0)

    t1 = time.time()
    outputs = model(img)
    t2 = time.time()
    print("Time for detection: " + str(t2 - t1))
    boxes = outputs[0]['boxes'].detach().numpy()
    scores = outputs[0]['scores'].detach().numpy()
    labels = outputs[0]['labels'].detach().numpy()

    crop_imgs = []
    crop_img_paths = []
    crop_img_classes = []
    img = cv2.imread(img_path + img_name)
    copy_img = copy.deepcopy(img)
    for i in range(len(boxes)):
        if scores[i] > 0.8:
            box = boxes[i]
            label = labels[i]
            [x1, y1, x2, y2] = [int(i) for i in box]
            color = list(np.random.choice(range(256), size=3))
            color = tuple([int(c) for c in color])
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label2class[label], (x1 - 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            crop_img = copy_img[y1:y2, x1:x2]
            crop_imgs.append(crop_img)
            crop_img_classes.append(label2class[label])

    output_name = img_name.replace('.', '_detected.')
    output_path = img_path + output_name
    cv2.imwrite(output_path, img)
    count = 1
    for crop_img in crop_imgs:
        output_name = img_name.replace('.', '_cropped_' + str(count) + '.')
        output_cropped_path = img_path + output_name
        crop_img_paths.append(output_cropped_path)
        cv2.imwrite(output_cropped_path, crop_img)
        count += 1

    return output_path, crop_img_paths, crop_img_classes


def bounding_box_image_with_pytorch(model, img_path):
    """
    Detect clothes of images, return only cropped images
    """
    p = img_path
    img = Image.open(p).convert("RGB")
    img = T.ToTensor()(img)
    img = torch.unsqueeze(img, 0)

    outputs = model(img)
    boxes = outputs[0]['boxes'].detach().numpy()
    scores = outputs[0]['scores'].detach().numpy()
    labels = outputs[0]['labels'].detach().numpy()

    crop_imgs = []
    img = cv2.imread(p)
    copy_img = copy.deepcopy(img)
    for i in range(len(boxes)):
        if scores[i] > 0.8:
            box = boxes[i]
            [x1, y1, x2, y2] = [int(i) for i in box]
            crop_img = copy_img[y1:y2, x1:x2]
            crop_imgs.append(crop_img)

    return crop_imgs

def detect_image(img_name, img_path, weight_path, config_path):
    """
    Detect clothes from the images, use pre-trained YOLO weight
    Return detected result and cropped images
    """
    net = get_model(config_path, weight_path)
    img = cv2.imread(img_path+img_name)
    copy_img = copy.deepcopy(img)

    (H, W) = img.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []
    classes = ['short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear',
                'vest', 'sling', 'shorts', 'trousers',
                'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress']

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            label = classes[classID]
            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height), label])
                confidences.append(float(confidence))
                classIDs.append(classID)
            
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    
    crop_imgs = []
    crop_img_paths = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            print(boxes)
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
            cv2.putText(img, boxes[i][4], (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            crop_img = copy_img[y:y+h, x:x+w]
            crop_imgs.append(crop_img)

    output_name = img_name.replace('.', '_detected.')
    output_path = img_path + output_name
    cv2.imwrite(output_path, img)
    count = 1
    for crop_img in crop_imgs:
        output_name = img_name.replace('.', '_cropped_' + str(count) + '.')
        output_cropped_path = img_path + output_name
        crop_img_paths.append(output_cropped_path)
        cv2.imwrite(output_cropped_path, crop_img)
        count += 1

    return crop_imgs


def bounding_box_image(net, img_path):
    """
    Return only image part in bounding box
    """
    # weight_path = os.path.join(app.config["IMAGE_UPLOADS"], 'models/yolov3-df2_15000.weights')
    # config_path = os.path.join(app.config["IMAGE_UPLOADS"], 'models/yolov3-df2.cfg')
    img = cv2.imread(img_path)
    copy_img = copy.deepcopy(img)

    (H, W) = img.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
            
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    
    crop_imgs = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            crop_img = copy_img[y:y+h, x:x+w]
            crop_img = cv2.resize(crop_img, (224, 224)) 
            crop_imgs.append(crop_img)

    return crop_imgs