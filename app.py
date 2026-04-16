from pathlib import Path
from uuid import uuid4

import base64
import os
import traceback

from flask import Flask, Response, jsonify, render_template, request

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
    files = [file for file in request.files.getlist("image") if file and file.filename]
    if not files:
        return jsonify({"success": False, "message": "Please choose one or more images to upload."}), 400

    invalid_name = next((file.filename for file in files if not allowed_file(file.filename)), None)
    if invalid_name:
        return jsonify(
            {
                "success": False,
                "message": f"Unsupported file type for '{invalid_name}'. Use PNG, JPG, JPEG, or WEBP.",
            }
        ), 400

    previous_paths = face_store.get_image_paths()
    saved_paths: list[Path] = []
    encodings = []

    try:
        for file in files:
            suffix = Path(file.filename).suffix.lower()
            save_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
            file.save(save_path)
            saved_paths.append(save_path)
            encodings.append(extract_face_encoding(save_path))
    except ValueError as exc:
        for path in saved_paths:
            path.unlink(missing_ok=True)
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        print(f"Failed to process upload batch: {exc!r}", flush=True)
        traceback.print_exc()
        for path in saved_paths:
            path.unlink(missing_ok=True)
        return jsonify({"success": False, "message": "Failed to process the uploaded image set."}), 500

    face_store.set_targets(encodings, [str(path) for path in saved_paths])

    current_paths = {str(path) for path in saved_paths}
    for previous_path in previous_paths:
        if previous_path not in current_paths:
            try:
                Path(previous_path).unlink(missing_ok=True)
            except Exception as exc:
                print(f"Failed to delete previous upload: {exc!r}", flush=True)

    target_count = face_store.get_target_count()
    return jsonify(
        {
            "success": True,
            "target_count": target_count,
            "message": (
                f"Loaded {target_count} target image(s). Using multiple clear reference images "
                "usually improves matching accuracy."
            ),
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
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
