import cv2
import numpy as np
import os
import time

def detect_image(img_name, img_path, weight_path, config_path):
    net = cv2.dnn.readNetFromDarknet(config_path, weight_path)
    img = cv2.imread(img_path+img_name)

    (H, W) = img.shape[:2]
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
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
    
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)

    output_name = img_name.replace('.', '_detected.')
    output_path = img_path + output_name
    cv2.imwrite(output_path, img)
    return output_path


def feature_image(img_path):
    return np.random.rand(1024)


def feature_all_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        # img = cv2.imread(os.path.join(folder_path,filename))
        img_path = os.path.join(folder_path, filename)
        img_feature = feature_image(img_path)
        if img_feature is not None:
            images.append({
                "img_path": img_path,
                "img_features": img_feature
            })
    return images
