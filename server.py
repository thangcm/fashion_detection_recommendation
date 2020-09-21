from flask import Flask, render_template, redirect, url_for, request
from controllers import detection, recommendation
import os
app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "static/"

@app.route("/", methods=["GET", "POST"])
def upload_image():
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
    filename = request.args.get("filename")
    weights_path = os.path.join(app.config["IMAGE_UPLOADS"], 'models/yolov3-df2_15000.weights')
    config_path = os.path.join(app.config["IMAGE_UPLOADS"], 'models/yolov3-df2.cfg')
    detected_image_path, crop_img_paths = detection.detect_image(filename, app.config["IMAGE_UPLOADS"], weights_path, config_path)
    
    recommend_images = []
    for path in crop_img_paths:
        recommend_images += recommendation.recommend(path)
        
    return render_template("show_detection.html", user_image = detected_image_path, results=recommend_images)


if __name__ == '__main__':
   app.run()
