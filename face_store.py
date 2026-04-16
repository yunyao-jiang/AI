from __future__ import annotations

from threading import Lock

import numpy as np


class FaceStore:
    """Thread-safe storage for one or more target face encodings."""

    def __init__(self) -> None:
        self._encodings: list[np.ndarray] = []
        self._image_paths: list[str] = []
        self._lock = Lock()

    def set_targets(self, encodings: list[np.ndarray], image_paths: list[str]) -> None:
        with self._lock:
            self._encodings = [np.array(encoding, dtype=np.float64) for encoding in encodings]
            self._image_paths = list(image_paths)

    def clear(self) -> None:
        with self._lock:
            self._encodings = []
            self._image_paths = []

    def get_targets(self) -> list[np.ndarray]:
        with self._lock:
            return [encoding.copy() for encoding in self._encodings]

    def get_image_paths(self) -> list[str]:
        with self._lock:
            return list(self._image_paths)

    def get_target_count(self) -> int:
        with self._lock:
            return len(self._encodings)

    def has_target(self) -> bool:
        with self._lock:
            return bool(self._encodings)


face_store = FaceStore()
