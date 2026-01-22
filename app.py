from flask import Flask, render_template, request, url_for
import os
from predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    explanation = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            result, confidence, explanation = predict_image(image_path)

            image_url = url_for("static", filename=f"uploads/{file.filename}")

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        explanation=explanation,
        image_url=image_url
    )

if __name__ == "__main__":
    app.run(debug=True)
