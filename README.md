# Real-Time Face Recognition Web App

A Flask-based face recognition demo that lets a user upload a reference face image and then compare live webcam frames against that target in the browser.

## Stack

- Python 3.11
- Flask
- OpenCV
- `face_recognition`

## Features

- Upload a target face image from the web UI
- Extract and store a 128-d face encoding
- Open the webcam in the browser
- Send browser frames to the Flask backend for analysis
- Detect faces in real time
- Label faces as `TARGET`, `OTHER`, or `NO TARGET`
- Draw live bounding boxes in the browser
- Output the matched target center `(x, y)` coordinates on the server side

## Project Structure

```text
.
├── app.py
├── face_store.py
├── recognizer_runtime.py
├── templates/
│   └── index.html
├── static/
└── uploads/
```

## Run Locally

Install dependencies:

```powershell
python -m pip install flask opencv-python face-recognition
```

Start the app:

```powershell
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## How It Works

1. Upload a clear image containing one visible face.
2. The backend saves the file to `uploads/` and computes a face encoding.
3. The browser opens the webcam with `getUserMedia`.
4. The frontend periodically sends compressed frames to `/analyze-frame`.
5. The backend detects faces, compares encodings against the stored target, and returns face metadata.
6. The frontend renders the live video and overlays the returned boxes and labels.

## Main Routes

- `GET /`  
  Main web UI

- `POST /upload`  
  Upload a target face image

- `POST /analyze-frame`  
  Analyze one browser-captured frame

- `GET /video`  
  MJPEG stream endpoint kept for OpenCV webcam mode support

## Notes

- The browser camera flow is used because it works more reliably than direct OpenCV webcam capture on many Windows setups.
- The current version stores one target reference image at a time.
- Accuracy depends heavily on lighting, pose, distance, and the quality of the uploaded reference photo.
- The app currently uses Flask's built-in server, which is fine for local development but not for production deployment.

## Known Limitations

- False positives are still possible when only one target image is used.
- Public deployment requires HTTPS because browser camera access does not work on normal insecure origins.
- `face_recognition` and `dlib` can be harder to install on newer Windows/Python combinations.

## Next Steps

- Support multiple target reference images
- Tighten matching with distance-based thresholds
- Add multi-frame confirmation for stronger target decisions
- Add Docker and production deployment configuration

## Deploy on Render

This repository now includes:

- `Dockerfile`
- `requirements.txt`
- `render.yaml`

To deploy on Render:

1. Push this repository to GitHub.
2. Sign in to [Render](https://render.com/).
3. Create a new Web Service from your GitHub repository.
4. Render should detect the Docker setup automatically.
5. Deploy the service and wait for the first build to finish.
6. Open the generated `https://...onrender.com` URL.

Notes:

- Public camera access requires `HTTPS`, so using a Render URL is important.
- Uploaded files are stored on ephemeral disk in `uploads/`, so they are not guaranteed to persist across redeploys or restarts.
- This app is currently configured as a demo service, not a hardened production system.
