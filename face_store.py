from __future__ import annotations

from threading import Lock
from typing import Optional

import numpy as np


class FaceStore:
    """Thread-safe storage for the currently selected target face."""

    def __init__(self) -> None:
        self._encoding: Optional[np.ndarray] = None
        self._image_path: Optional[str] = None
        self._lock = Lock()

    def set_target(self, encoding: np.ndarray, image_path: str) -> None:
        with self._lock:
            self._encoding = np.array(encoding, dtype=np.float64)
            self._image_path = image_path

    def clear(self) -> None:
        with self._lock:
            self._encoding = None
            self._image_path = None

    def get_target(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._encoding is None:
                return None
            return self._encoding.copy()

    def get_image_path(self) -> Optional[str]:
        with self._lock:
            return self._image_path

    def has_target(self) -> bool:
        with self._lock:
            return self._encoding is not None


face_store = FaceStore()
