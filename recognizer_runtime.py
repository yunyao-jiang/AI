from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Generator, Optional

import cv2
import face_recognition
import numpy as np

from face_store import face_store


def extract_face_encoding(image_path: str | Path) -> np.ndarray:
    image = face_recognition.load_image_file(str(image_path))
    face_locations = face_recognition.face_locations(image, model="hog")

    if not face_locations:
        raise ValueError("No face detected in the uploaded image.")

    encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
    if not encodings:
        raise ValueError("Could not generate a face encoding from the uploaded image.")

    return encodings[0]


class FaceRecognizer:
    def __init__(self, video_source: int = 0, resize_scale: float = 0.5, tolerance: float = 0.5) -> None:
        self.video_source = video_source
        self.resize_scale = resize_scale
        self.tolerance = tolerance

    def generate_frames(self) -> Generator[bytes, None, None]:
        camera = self._open_camera()

        if camera is None or not camera.isOpened():
            yield from self._error_frame_stream(
                "No camera available. Check Windows camera privacy, close Zoom/Teams, or connect a webcam."
            )
            return

        try:
            while True:
                ok, frame = camera.read()
                if not ok or frame is None:
                    yield self._encode_frame(self._build_message_frame("Failed to read from webcam."))
                    time.sleep(0.2)
                    continue

                processed_frame, _ = self.process_frame_with_metadata(frame)
                yield self._encode_frame(processed_frame)
        finally:
            camera.release()

    def process_frame_with_metadata(self, frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        display_frame = frame.copy()
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=self.resize_scale,
            fy=self.resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        metadata: dict[str, Any] = {
            "target_loaded": face_store.has_target(),
            "target_count": face_store.get_target_count(),
            "faces": [],
            "target_center": None,
        }

        if not face_locations:
            self._draw_banner(display_frame, "No face detected", (0, 165, 255))
            return display_frame, metadata

        face_encodings = face_recognition.face_encodings(rgb_small_frame, known_face_locations=face_locations)
        target_encodings = face_store.get_targets()

        if not target_encodings:
            self._draw_banner(display_frame, "Upload target face images to start matching", (0, 165, 255))

        scale_back = int(round(1 / self.resize_scale))

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= scale_back
            right *= scale_back
            bottom *= scale_back
            left *= scale_back

            is_match = False
            label = "OTHER"
            color = (0, 0, 255)

            if target_encodings:
                distances = face_recognition.face_distance(target_encodings, face_encoding)
                is_match = bool(len(distances) and float(np.min(distances)) <= self.tolerance)
            else:
                label = "NO TARGET"
                color = (0, 165, 255)

            if is_match:
                label = "TARGET"
                color = (0, 255, 0)
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
                metadata["target_center"] = {"x": center_x, "y": center_y}
                print(f"TARGET center: ({center_x}, {center_y})", flush=True)

            metadata["faces"].append(
                {
                    "label": label,
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                }
            )
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            self._draw_label(display_frame, label, left, top, color)

        return display_frame, metadata

    def analyze_encoded_frame(self, payload: bytes) -> dict[str, Any]:
        frame_array = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode the submitted frame.")

        _, metadata = self.process_frame_with_metadata(frame)
        metadata["frame_width"] = int(frame.shape[1])
        metadata["frame_height"] = int(frame.shape[0])
        return metadata

    def _error_frame_stream(self, message: str) -> Generator[bytes, None, None]:
        while True:
            frame = self._build_message_frame(message)
            yield self._encode_frame(frame)
            time.sleep(0.5)

    def _open_camera(self) -> Optional[cv2.VideoCapture]:
        attempts = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("DEFAULT", None)]

        for backend_name, backend in attempts:
            camera = cv2.VideoCapture(self.video_source) if backend is None else cv2.VideoCapture(self.video_source, backend)
            if camera.isOpened():
                print(f"Opened webcam with backend: {backend_name}", flush=True)
                return camera
            camera.release()

        return None

    @staticmethod
    def _build_message_frame(message: str) -> np.ndarray:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, message, (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    @staticmethod
    def _draw_banner(frame: np.ndarray, message: str, color: tuple[int, int, int]) -> None:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), color, -1)
        cv2.putText(frame, message, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _draw_label(frame: np.ndarray, text: str, left: int, top: int, color: tuple[int, int, int]) -> None:
        label_top = max(top - 30, 0)
        cv2.rectangle(frame, (left, label_top), (left + 140, top), color, -1)
        cv2.putText(frame, text, (left + 8, max(top - 8, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> bytes:
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            fallback = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(fallback, "Frame encode failed", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            _, buffer = cv2.imencode(".jpg", fallback)

        return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
