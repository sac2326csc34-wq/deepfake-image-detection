from flask import Flask, render_template, request, url_for, redirect, flash
import os
from werkzeug.utils import secure_filename
from predict import predict_image

app = Flask(__name__)
app.secret_key = "deepfake_secret_key"

# ----------------------------
# Upload Configuration
# ----------------------------

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ----------------------------
# Routes
# ----------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    print("INDEX ROUTE HIT")

    result = None
    confidence = None
    explanation = None
    image_url = None
    boxed_url = None

    if request.method == "POST":
        print("FORM SUBMITTED")

        if "image" not in request.files:
            flash("No file selected.")
            return redirect(request.url)

        file = request.files["image"]

        if file.filename == "":
            flash("Please select an image.")
            return redirect(request.url)

        if file and allowed_file(file.filename):

            filename = secure_filename(file.filename)

            # Prevent overwriting files
            base, ext = os.path.splitext(filename)
            counter = 1
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

            while os.path.exists(image_path):
                filename = f"{base}_{counter}{ext}"
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                counter += 1

            file.save(image_path)

            # 🔥 Run prediction (3 values only)
            result, confidence, explanation, boxed_path = predict_image(image_path)

            boxed_filename = os.path.basename(boxed_path)
            boxed_url = url_for("static", filename=f"uploads/{boxed_filename}")
            image_url = url_for("static", filename=f"uploads/{filename}")

        else:
            flash("Invalid file type. Only PNG, JPG, JPEG allowed.")
            return redirect(request.url)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        explanation=explanation,
        image_url=image_url,
        boxed_url=boxed_url
    )

@app.route("/test")
def test():
    return "Server working"

if __name__ == "__main__":
    print("THIS IS THE ACTIVE APP FILE")
    app.run(debug=True)