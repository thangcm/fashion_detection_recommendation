from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import numpy as np
import os
from controllers import detection
from server import app
import cv2

# model = VGG16(weights='imagenet', include_top=False)
model = ResNet50(weights='imagenet', include_top=False)
model.summary()

def feature_image(img=None, img_path=None):
    """
    Return features vector of image
    :param img:
    :param img_path:
    :return:
    """
    if img is None:
        img = image.load_img(img_path, target_size=(128, 128))
    else:
        img = cv2.resize(img, (128, 128))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    feature = model.predict(img_data)
    feature_np = np.array(feature)
    print(feature_np.shape)
    return feature_np.flatten()


def feature_all_images(folder_path):
    """
    Return all feature vector of images in the specific directory
    """
    images = []
    for filename in os.listdir(folder_path):
        # img = cv2.imread(os.path.join(folder_path,filename))
        img_path = os.path.join(folder_path, filename)
        img_feature = feature_image(img_path=img_path)
        if img_feature is not None:
            images.append({
                "img_path": img_path,
                "img_features": img_feature
            })
    return images


def feature_all_images_bounding_box(folder_path):
    """
    Return feature vector of bounding box of all images in the directory
    1. Detect clothes of image and get cropped images using YOLO
    2. Get the feature vector of cropped images
    """
    images = []
    weight_path = os.path.join(app.config["IMAGE_UPLOADS"], 'models/yolov3-df2_15000.weights')
    config_path = os.path.join(app.config["IMAGE_UPLOADS"], 'models/yolov3-df2.cfg')
    net = detection.get_model(config_path, weight_path)
    for filename in os.listdir(folder_path):
        # img = cv2.imread(os.path.join(folder_path,filename))
        img_path = os.path.join(folder_path, filename)
        cropped_imgs = detection.bounding_box_image(net, img_path)
        for cropped_img in cropped_imgs:
            img_feature = feature_image(img=cropped_img)
            if img_feature is not None:
                images.append({
                    "img_path": img_path,
                    "img_features": img_feature
                })
    return images


def feature_all_images_bounding_box_pytorch(folder_path):

    """
    Return feature vector of bounding box of all images in the directory
    1. Detect clothes of image and get cropped images using Pytorch model
    2. Get the feature vector of cropped images
    """
    images = []
    weight_path = os.path.join(app.config["IMAGE_UPLOADS"], 'models/clothing_detection.pth')
    net = detection.get_model_pytorch(weight_path)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        cropped_imgs = detection.bounding_box_image_with_pytorch(net, img_path)
        for cropped_img in cropped_imgs:
            img_feature = feature_image(img=cropped_img)
            if img_feature is not None:
                images.append({
                    "img_path": img_path,
                    "img_features": img_feature
                })
    return images