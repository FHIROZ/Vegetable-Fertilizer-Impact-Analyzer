from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(
    __name__,
    static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "static")),
    static_url_path="/static",
)

# Resolve uploads directory relative to workspace root and ensure it exists
UPLOAD_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model from the project directory with explicit path and clear error on failure
model_path = os.path.join(os.path.dirname(__file__), "vegetable_fertilizer_classifier.h5")
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Failed to load model from {model_path}: {e}")
    raise


classes = ["healthy", "nutrient_deficient", "over_fertilized"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]

    filename = secure_filename(file.filename)
    if filename == "":
        return jsonify({"error": "No file selected or invalid filename"})

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Load & preprocess image
    img = image.load_img(filepath, target_size=(300, 300))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img)[0]
    idx = np.argmax(preds)

    label = classes[idx]
    confidence = round(float(preds[idx]) * 100, 2)

    # Business logic
    if label == "healthy":
        fertilizer = "Balanced fertilizer usage"
        organic = "High"
    elif label == "nutrient_deficient":
        fertilizer = "Low fertilizer impact"
        organic = "Medium"
    else:
        fertilizer = "High fertilizer impact"
        organic = "Low"

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "fertilizer": fertilizer,
        "organic": organic
    })

if __name__ == "__main__":
    app.run(debug=True)
