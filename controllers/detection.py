import cv2
import numpy as np
import os
import time
import copy

def detect_image(img_name, img_path, weight_path, config_path):
    """
    Detect clothes from the images, use pre-trained YOLO weight
    Return detected result and cropped images
    """
    net = cv2.dnn.readNetFromDarknet(config_path, weight_path)
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

    return output_path, crop_img_paths