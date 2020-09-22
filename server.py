from flask import Flask, render_template, redirect, url_for, request
from controllers import detection, recommendation
import os
import time
app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "static/"

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
  
    return render_template("show_detection.html", user_image = detected_image_path, results=zip(crop_img_paths, recommend_images))


if __name__ == '__main__':
   app.run()
