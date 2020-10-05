from flask import Flask, render_template, redirect, url_for, request
from controllers import detection
import os
from lshashpy3 import LSHash
app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "static/"
app.config["IMAGE_STORAGE"] = "static/imgs"

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
    detected_image_path = detection.detect_image(filename, app.config["IMAGE_UPLOADS"], weights_path, config_path)
    return render_template("show_detection.html", user_image = detected_image_path)


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
