"""
Flask Application
==================
Serves the CycleGAN web interface for face ↔ sketch translation.

Routes
------
  GET  /                  → main UI page
  POST /translate         → translate an uploaded image (multipart/form-data)
  POST /translate_base64  → translate a base64-encoded image (JSON)
                            used for live camera input from the browser
  GET  /health            → health-check / model status
"""

import base64
import io
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

from inference import (
    bytes_to_pil,
    detect_domain,
    load_generators,
    pil_to_bytes,
    translate_image,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit

# Path to your trained checkpoint – set via env var or default
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "./checkpoints/latest.pth")

_models_ready = False


def init_models():
    global _models_ready
    if not os.path.isfile(CHECKPOINT_PATH):
        print(
            f"[WARNING] Checkpoint not found at: {CHECKPOINT_PATH}\n"
            f"          Set the CHECKPOINT_PATH environment variable or "
            f"train the model first.\n"
            f"          The server will start but /translate will return errors."
        )
        return
    try:
        load_generators(CHECKPOINT_PATH)
        _models_ready = True
        print("[Flask] Models ready ✓")
    except Exception as exc:
        print(f"[Flask] Failed to load models: {exc}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", models_ready=_models_ready)


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_ready": _models_ready,
        "checkpoint": CHECKPOINT_PATH,
    })


@app.route("/translate", methods=["POST"])
def translate():
    """
    Accepts: multipart/form-data
        file      : image file (required)
        direction : 'auto' | 'face2sketch' | 'sketch2face'  (optional)
    Returns: PNG image
    """
    if not _models_ready:
        return jsonify({"error": "Models not loaded. Please train first."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use key 'file'."}), 400

    f = request.files["file"]
    direction = request.form.get("direction", "auto")

    try:
        pil_image = Image.open(f).convert("RGB")
        output, actual_dir = translate_image(pil_image, direction)
        img_bytes = pil_to_bytes(output, fmt="PNG")
        label = "face → sketch" if actual_dir == "face2sketch" else "sketch → face"
        response = send_file(
            io.BytesIO(img_bytes),
            mimetype="image/png",
            as_attachment=False,
        )
        response.headers["X-Translation-Direction"] = label
        return response

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/translate_base64", methods=["POST"])
def translate_base64():
    """
    Accepts: JSON  { "image": "<base64-encoded PNG/JPEG>",
                     "direction": "auto" }
    Returns: JSON  { "image": "<base64 PNG>",
                     "direction": "face → sketch" | "sketch → face",
                     "detected_domain": "face" | "sketch" }
    """
    if not _models_ready:
        return jsonify({"error": "Models not loaded. Please train first."}), 503

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "JSON body must contain 'image' key."}), 400

    direction = data.get("direction", "auto")

    try:
        # Strip data-URL prefix if present
        b64 = data["image"]
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        pil_image = bytes_to_pil(img_bytes)

        detected = detect_domain(pil_image)
        output, actual_dir = translate_image(pil_image, direction)
        out_bytes = pil_to_bytes(output, fmt="PNG")
        out_b64 = base64.b64encode(out_bytes).decode()

        label = "face → sketch" if actual_dir == "face2sketch" else "sketch → face"
        return jsonify({
            "image":           f"data:image/png;base64,{out_b64}",
            "direction":       label,
            "detected_domain": detected,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_models()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
