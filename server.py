from flask import Flask, render_template, redirect, url_for, request
from controllers import detection, recommendation
from utils import feature_extraction
import os
from lshashpy3 import LSHash
import time
import pickle

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "static/"
app.config["IMAGE_STORAGE"] = "static/imgs"


@app.route("/", methods=["GET", "POST"])
def upload_image():
    """
    A view prepared for upload images
    Images will be save in /static folder
    """
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            if image.filename == "":
                print("No filename")
                return redirect(request.url + "detection?filename=" + filename)

            filename = image.filename
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))
            print("Image saved")
            return redirect(request.url + "detection?filename=" + filename)

    return render_template("upload_image.html")


@app.route("/detection", methods=["GET"])
def detect_image():
    """
    A view show detection and recommendation results, includes:
    - Image with bounding boxes and label
    - Cropped images
    - Recommended images
    """
    filename = request.args.get("filename")
    weights_path = os.path.join(app.config["IMAGE_UPLOADS"], 'models/clothing_detection.pth')
    # weights_path = os.path.join(app.config["IMAGE_UPLOADS"], 'models/yolov3-df2_15000.weights')
    config_path = os.path.join(app.config["IMAGE_UPLOADS"], 'models/yolov3-df2.cfg')

    detected_image_path, crop_img_paths, crop_img_classes = detection.detect_image_with_pytorch(filename, app.config["IMAGE_UPLOADS"], weights_path, config_path)


    t1 = time.time()
    recommend_images = []
    for path in crop_img_paths:
        recommend_images.append(recommendation.recommend_heapq(path))
        # recommend_images.append(recommendation.recommend_img(path))


    t2 = time.time()
    print("Time for recommendation: " + str(t2 - t1))
    # return render_template("show_detection.html", user_image = detected_image_path)
  
    return render_template("show_results.html", user_image = detected_image_path, results=zip(crop_img_paths, recommend_images, crop_img_classes))


def create_hash_table():
    """
    Save and store the features of images in:
    - Pickle file
    - Locality sensitive hashing
    """
    # features = feature_extraction.feature_all_images_bounding_box_pytorch(app.config["IMAGE_STORAGE"])
    features = feature_extraction.feature_all_images(app.config["IMAGE_STORAGE"])
    pickle.dump(features, open("feature_dict.p", "wb"))

    # Only used for big dataset
    # lsh = LSHash(hash_size=10, input_dim=32768, num_hashtables=3,
    #     storage_config={ 'dict': None },
    #     matrices_filename='controllers/weights.npz',
    #     hashtable_filename='controllers/hash.npz',
    #     overwrite=True
    #     )
    # for feature in features:
    #     lsh.index(feature["img_features"], extra_data=feature["img_path"])
    # lsh.save()


if __name__ == '__main__':
    create_hash_table()
    app.run()