from flask import Flask, render_template, redirect, url_for, request
from controllers import detection, recommendation
import os
<<<<<<< HEAD
from lshashpy3 import LSHash
=======
import time
>>>>>>> 3630a332e9e3d7d490917680a5e86e4a748da2bd
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
    weights_path = os.path.join(app.config["IMAGE_UPLOADS"], 'models/yolov3-df2_15000.weights')
    config_path = os.path.join(app.config["IMAGE_UPLOADS"], 'models/yolov3-df2.cfg')

    t1= time.time()
    detected_image_path, crop_img_paths = detection.detect_image(filename, app.config["IMAGE_UPLOADS"], weights_path, config_path)
    t2 = time.time()
    print("Time for detection: " + str(t2 - t1))

    t1 = time.time()
    recommend_images = []
    for path in crop_img_paths:
        recommend_images.append(recommendation.recommend(path))
    t2 = time.time()
    print("Time for recommendation: " + str(t2 - t1))
    # return render_template("show_detection.html", user_image = detected_image_path)
  
    return render_template("show_results.html", user_image = detected_image_path, results=zip(crop_img_paths, recommend_images))


def create_hash_table():
    features = detection.feature_all_images(app.config["IMAGE_STORAGE"])
    lsh = LSHash(
        10, 1024, storage_config={ 'dict': None },
        matrices_filename='weights.npz',
        hashtable_filename='hash.npz',
        overwrite=True
        )
    for feature in features:
        lsh.index(feature["img_features"], extra_data=feature["img_path"])

    lsh.save()
    print(lsh)

if __name__ == '__main__':
    create_hash_table()
    app.run()
