from pathlib import Path
from uuid import uuid4

import base64
import traceback

from flask import Flask, Response, jsonify, render_template, request
from werkzeug.utils import secure_filename

from face_store import face_store
from recognizer_runtime import FaceRecognizer, extract_face_encoding


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)

recognizer = FaceRecognizer(video_source=0, resize_scale=0.5)


@app.after_request
def disable_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"success": False, "message": "Uploaded image is too large. Please use a file under 8 MB."}), 413


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    print(f"Unhandled server error: {error!r}", flush=True)
    traceback.print_exc()
    return jsonify({"success": False, "message": "Server error while processing the request."}), 500


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image file was provided."}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"success": False, "message": "Please choose an image to upload."}), 400

    if not allowed_file(file.filename):
        return jsonify(
            {"success": False, "message": "Unsupported file type. Use PNG, JPG, JPEG, or WEBP."}
        ), 400

    original_name = file.filename
    suffix = Path(original_name).suffix.lower()
    stored_name = f"{uuid4().hex}{suffix}"
    save_path = UPLOAD_DIR / stored_name
    try:
        file.save(save_path)
    except Exception as exc:
        print(f"Failed to save upload: {exc!r}", flush=True)
        return jsonify({"success": False, "message": "Failed to save the uploaded image."}), 500

    try:
        encoding = extract_face_encoding(save_path)
    except ValueError as exc:
        save_path.unlink(missing_ok=True)
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        print(f"Failed to process upload: {exc!r}", flush=True)
        traceback.print_exc()
        save_path.unlink(missing_ok=True)
        return jsonify({"success": False, "message": "Failed to process the uploaded image."}), 500

    previous_path = face_store.get_image_path()
    face_store.set_target(encoding, str(save_path))

    if previous_path and previous_path != str(save_path):
        try:
            Path(previous_path).unlink(missing_ok=True)
        except Exception as exc:
            print(f"Failed to delete previous upload: {exc!r}", flush=True)

    return jsonify(
        {
            "success": True,
            "message": "Target face loaded successfully. Real-time matching is now active.",
        }
    )


@app.route("/video")
def video():
    return Response(
        recognizer.generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/analyze-frame", methods=["POST"])
def analyze_frame():
    frame_file = request.files.get("frame")
    frame_data = request.form.get("frame")

    if frame_file is not None and frame_file.filename:
        payload = frame_file.read()
    elif frame_data:
        if "," in frame_data:
            _, encoded = frame_data.split(",", 1)
        else:
            encoded = frame_data
        try:
            payload = base64.b64decode(encoded)
        except Exception:
            return jsonify({"success": False, "message": "Invalid frame payload."}), 400
    else:
        return jsonify({"success": False, "message": "No frame was provided."}), 400

    try:
        result = recognizer.analyze_encoded_frame(payload)
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        print(f"Frame analysis error: {exc!r}", flush=True)
        traceback.print_exc()
        return jsonify({"success": False, "message": "Failed to analyze the frame."}), 500

    result["success"] = True
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
