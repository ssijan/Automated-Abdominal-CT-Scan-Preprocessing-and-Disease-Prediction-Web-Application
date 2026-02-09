from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    url_for,
    send_from_directory,
)
from pathlib import Path
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

from PIL import Image
import numpy as np
import json
import cv2
from typing import Optional

# ---------- PyTorch (ProtoNet) ----------
# ProtoNet (PyTorch) support removed due to Torch errors


# ----------------- FLASK SETUP -----------------

app = Flask(__name__)
BASE_DIR = Path(__file__).parent

UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

CLEANED_FOLDER = BASE_DIR / "uploads_cleaned"
CLEANED_FOLDER.mkdir(exist_ok=True)

MODELS_JSON_PATH = BASE_DIR / "models.json"

print("TF version:", tf.__version__)
print("Torch support: disabled in this app (PyTorch code removed)")


# ----------------- DASHBOARD MODEL METADATA -----------------

def load_models():
    with MODELS_JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

@app.route("/api/models")
def list_models():
    models = load_models()
    resp = []
    for mid, meta in models.items():
        resp.append({
            "id": mid,
            "name": meta.get("name", mid),
            "description": meta.get("description", ""),
        })
    return jsonify(resp)

@app.route("/api/models/<model_id>")
def get_model_metadata(model_id):
    models = load_models()
    if model_id not in models:
        return jsonify({"error": "Model not found"}), 404
    return jsonify(models[model_id])


# ----------------- SHARED SETTINGS -----------------

IMAGE_SIZE = (224, 224, 3)
NUM_CLASSES = 11

CLASS_NAMES = [
    "Abdominal_Wall_Hernia",
    "Bulky_Uterus",
    "Duodenum_Mass",
    "Hepatic_Mass",
    "Liver_SOL",
    "Metastatic_Adenocarcinoma",
    "Metastatic_Lesions",
    "Multicentric_HCC",
    "Normal",
    "Ovary_Cyst",
    "Renal_Cortical_Cysts",
]

model_cache = {}


# ----------------- TF MODELS (MobileNet / ResNet) -----------------

def build_mobilenet_model():
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=IMAGE_SIZE)
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.6),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.build((None,) + IMAGE_SIZE)

    weights_path = BASE_DIR / "models" / "mobilenet" / "mobilenet_model.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"MobileNetV2 weights file not found: {weights_path}")

    model.load_weights(weights_path)
    print("MobileNetV2 weights loaded:", weights_path)
    return model


def build_resnet_model():
    base_model = ResNet50(weights=None, include_top=False, input_shape=IMAGE_SIZE)
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu", kernel_regularizer=l2(0.001)),
        Dropout(0.6),
        Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.build((None,) + IMAGE_SIZE)

    weights_path = BASE_DIR / "models" / "resnet50" / "model.weights.h5"
    if not weights_path.exists():
        raise FileNotFoundError(f"ResNet50 weights file not found: {weights_path}")

    model.load_weights(weights_path)
    print("ResNet50 weights loaded:", weights_path)
    return model


# ----------------- ProtoNet (Few-shot) -----------------
# IMPORTANT:
# ProtoNet needs class "prototypes" (support embeddings). We compute them from a folder:
#   models/protonet/support/<class_name>/*.jpg
# Put a few images (e.g., 3~10) per class in that support folder.

# ProtoNet support disabled. If you want to re-enable later,
# restore the PyTorch imports and related model/build code.


def get_model(model_id: str):
    if model_id in model_cache:
        return model_cache[model_id]

    if model_id == "mobilenetv2":
        m = build_mobilenet_model()
    elif model_id == "resnet50":
        m = build_resnet_model()
    else:
        m = None

    model_cache[model_id] = m
    return m


# ----------------- TEXT REMOVAL (OpenCV) -----------------

IMG_REF_SIZE = 512
BASE_RADIUS = 250
EDGE_RING_W = 35
TOP_FRACTION = 0.24

LOWER_YELLOW = np.array([5, 30, 110], dtype=np.uint8)
UPPER_YELLOW = np.array([55, 255, 255], dtype=np.uint8)
LOWER_GREEN = np.array([40, 30, 110], dtype=np.uint8)
UPPER_GREEN = np.array([90, 255, 255], dtype=np.uint8)
LOWER_WHITE = np.array([0, 0, 190], dtype=np.uint8)
UPPER_WHITE = np.array([180, 60, 255], dtype=np.uint8)


def remove_text_overlays(input_path: Path, output_path: Path) -> Optional[Path]:
    """
    Remove colored overlay text and save to output_path.
    Then paint 3px black on left and bottom edges.
    """
    img = cv2.imread(str(input_path))
    if img is None:
        print(f"[WARN] Could not read {input_path}")
        return None

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    radius = int(min(h, w) * (BASE_RADIUS / IMG_REF_SIZE))
    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2

    inside_circle = dist_sq <= radius**2
    outside_circle = ~inside_circle

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    mask_green  = cv2.inRange(hsv, LOWER_GREEN,  UPPER_GREEN)
    mask_white  = cv2.inRange(hsv, LOWER_WHITE,  UPPER_WHITE)
    color_mask = mask_yellow | mask_green | mask_white

    inner_r2 = (radius - EDGE_RING_W) ** 2
    edge_ring = (dist_sq <= radius**2) & (dist_sq >= inner_r2)

    top_rows = int(h * TOP_FRACTION)
    top_band = np.zeros((h, w), dtype=bool)
    top_band[:top_rows, :] = True

    bottom_and_sides = outside_circle

    region_mask = edge_ring | top_band | bottom_and_sides
    text_mask = (color_mask > 0) & region_mask

    kernel = np.ones((3, 3), np.uint8)
    text_mask_u8 = np.zeros((h, w), dtype=np.uint8)
    text_mask_u8[text_mask] = 255
    text_mask_u8 = cv2.dilate(text_mask_u8, kernel, iterations=3)

    cleaned = cv2.inpaint(img, text_mask_u8, 3, cv2.INPAINT_TELEA)

    # 3px black border
    cleaned[:, 0:3] = 0
    cleaned[h-3:h, :] = 0

    cv2.imwrite(str(output_path), cleaned)
    return output_path


# ----------------- IMAGE UTILS -----------------

def load_image(file_or_path, target_size=(224, 224)):
    if hasattr(file_or_path, "stream"):
        img = Image.open(file_or_path.stream).convert("RGB")
    else:
        img = Image.open(file_or_path).convert("RGB")

    img = img.resize(target_size)
    x = np.array(img).astype("float32")
    x = np.expand_dims(x, axis=0)  # (1,H,W,3) in 0..255
    return x


def preprocess_for_tf_model(x, model_id: str):
    if model_id == "mobilenetv2":
        return mobilenet_preprocess(x.copy())
    elif model_id == "resnet50":
        return resnet_preprocess(x.copy())
    else:
        return x / 255.0


# ProtoNet prediction removed. Torch-based few-shot predictions are disabled.


# ----------------- ROUTES -----------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")

@app.route("/cleaned/<filename>")
def get_cleaned_image(filename):
    return send_from_directory(str(CLEANED_FOLDER), filename)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    model_id = request.form.get("model_id", "mobilenetv2")

    filename = secure_filename(file.filename)
    save_path = UPLOAD_FOLDER / filename
    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({"error": "Could not save uploaded file", "details": str(e)}), 500

    # cleaned
    cleaned_filename = f"cleaned_{filename}"
    cleaned_path = CLEANED_FOLDER / cleaned_filename
    cleaned_ok = remove_text_overlays(save_path, cleaned_path)

    used_path = cleaned_path if (cleaned_ok is not None and cleaned_path.exists()) else save_path

    # pick model
    selected = get_model(model_id)
    if selected is None:
        return jsonify({
            "error": f"Model '{model_id}' is not supported.",
            "supported_models": ["mobilenetv2", "resnet50"],
        }), 400

    try:
        img_raw = load_image(used_path, target_size=(224, 224))
        img_batch = preprocess_for_tf_model(img_raw, model_id=model_id)
        preds = selected.predict(img_batch)[0]  # (11,) 0..1

        class_idx = int(np.argmax(preds))
        conf = float(np.max(preds))

        resp = {
            "predicted_class": CLASS_NAMES[class_idx],
            "confidence": conf,          # 0..1 (frontend converts to %)
            "probs": preds.tolist(),     # 0..1
            "classes": CLASS_NAMES,
            "model_id": model_id,
            "saved_file": filename,
        }

        if cleaned_ok is not None and cleaned_path.exists():
            resp["cleaned_file"] = cleaned_filename
            resp["cleaned_url"] = url_for("get_cleaned_image", filename=cleaned_filename)

        return jsonify(resp)

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e),
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
