"""
Drowsiness Detection Engine
Uses MediaPipe FaceLandmarker (tasks API) to compute EAR, MAR, and head tilt
for real-time drowsiness detection.
"""

import math
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ── Model path ───────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

# ── MediaPipe Face Mesh landmark indices (468-point mesh) ────────────────────

# Right eye (6 points for EAR)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Left eye (6 points for EAR)
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Mouth landmarks for MAR
MOUTH_VERTICAL = [
    (81, 178),   # inner lip left-vertical pair
    (13, 14),    # center vertical pair
    (311, 402),  # inner lip right-vertical pair
]
MOUTH_HORIZONTAL = (78, 308)  # left corner, right corner

# Head pose estimation landmarks
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 263
RIGHT_EYE_CORNER = 33
LEFT_MOUTH_CORNER = 308
RIGHT_MOUTH_CORNER = 78


@dataclass
class DetectionResult:
    """Result from a single frame's drowsiness analysis."""
    face_detected: bool = False
    ear: float = 0.0
    mar: float = 0.0
    head_tilt: float = 0.0
    eyes_closed: bool = False
    yawning: bool = False
    head_drooping: bool = False
    drowsy: bool = False
    alert_level: int = 0       # 0=awake, 1=warning, 2=drowsy
    landmarks: Optional[list] = field(default=None, repr=False)


class DrowsinessDetector:
    """Real-time drowsiness detector using MediaPipe FaceLandmarker."""

    # ── Tunable thresholds ────────────────────────────────────────────────
    EAR_THRESHOLD = 0.22
    MAR_THRESHOLD = 0.75
    HEAD_TILT_THRESHOLD = 30.0
    CONSEC_FRAMES_WARNING = 10
    CONSEC_FRAMES_DROWSY = 20
    YAWN_CONSEC_FRAMES = 15

    def __init__(self):
        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        self._eye_closed_counter = 0
        self._yawn_counter = 0
        self._head_droop_counter = 0

    # ── Public API ────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> DetectionResult:
        """Analyse a BGR frame and return drowsiness metrics."""
        result = DetectionResult()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self.landmarker.detect(mp_image)

        if not detection.face_landmarks:
            return result

        face_lm = detection.face_landmarks[0]
        h, w, _ = frame.shape
        landmarks = [(lm.x * w, lm.y * h) for lm in face_lm]

        result.face_detected = True
        result.landmarks = landmarks

        # Eye Aspect Ratio
        ear_right = self._compute_ear(landmarks, RIGHT_EYE)
        ear_left = self._compute_ear(landmarks, LEFT_EYE)
        result.ear = (ear_right + ear_left) / 2.0

        # Mouth Aspect Ratio
        result.mar = self._compute_mar(landmarks)

        # Head tilt (pitch)
        result.head_tilt = self._compute_head_tilt(landmarks, w, h)

        # ── Drowsiness logic ──────────────────────────────────────────────

        # Eyes closed tracking
        if result.ear < self.EAR_THRESHOLD:
            self._eye_closed_counter += 1
            result.eyes_closed = True
        else:
            self._eye_closed_counter = 0
            result.eyes_closed = False

        # Yawn tracking
        if result.mar > self.MAR_THRESHOLD:
            self._yawn_counter += 1
            result.yawning = True
        else:
            self._yawn_counter = 0
            result.yawning = False

        # Head droop tracking
        if abs(result.head_tilt) > self.HEAD_TILT_THRESHOLD:
            self._head_droop_counter += 1
            result.head_drooping = True
        else:
            self._head_droop_counter = 0
            result.head_drooping = False

        # Alert level determination
        drowsy_signals = 0
        if self._eye_closed_counter >= self.CONSEC_FRAMES_DROWSY:
            drowsy_signals += 2
        elif self._eye_closed_counter >= self.CONSEC_FRAMES_WARNING:
            drowsy_signals += 1

        if self._yawn_counter >= self.YAWN_CONSEC_FRAMES:
            drowsy_signals += 1

        if self._head_droop_counter >= self.CONSEC_FRAMES_WARNING:
            drowsy_signals += 1

        if drowsy_signals >= 2:
            result.alert_level = 2
            result.drowsy = True
        elif drowsy_signals >= 1:
            result.alert_level = 1
        else:
            result.alert_level = 0
            result.drowsy = False

        # Eye closure alone for enough frames is always drowsy
        if self._eye_closed_counter >= self.CONSEC_FRAMES_DROWSY:
            result.alert_level = 2
            result.drowsy = True

        return result

    def release(self):
        """Release MediaPipe resources."""
        self.landmarker.close()

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _dist(p1: tuple, p2: tuple) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _compute_ear(self, landmarks: list, eye_indices: list) -> float:
        """
        Eye Aspect Ratio.
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        p = [landmarks[i] for i in eye_indices]
        vertical_1 = self._dist(p[1], p[5])
        vertical_2 = self._dist(p[2], p[4])
        horizontal = self._dist(p[0], p[3])
        if horizontal == 0:
            return 0.0
        return (vertical_1 + vertical_2) / (2.0 * horizontal)

    def _compute_mar(self, landmarks: list) -> float:
        """Mouth Aspect Ratio = avg(vertical) / horizontal."""
        vert_sum = 0.0
        for top_idx, bot_idx in MOUTH_VERTICAL:
            vert_sum += self._dist(landmarks[top_idx], landmarks[bot_idx])
        vert_avg = vert_sum / len(MOUTH_VERTICAL)

        horiz = self._dist(
            landmarks[MOUTH_HORIZONTAL[0]],
            landmarks[MOUTH_HORIZONTAL[1]],
        )
        if horiz == 0:
            return 0.0
        return vert_avg / horiz

    def _compute_head_tilt(self, landmarks: list, w: int, h: int) -> float:
        """Estimate head pitch angle using solvePnP. Returns degrees."""
        model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ], dtype=np.float64)

        image_points = np.array([
            landmarks[NOSE_TIP],
            landmarks[CHIN],
            landmarks[LEFT_EYE_CORNER],
            landmarks[RIGHT_EYE_CORNER],
            landmarks[LEFT_MOUTH_CORNER],
            landmarks[RIGHT_MOUTH_CORNER],
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, _ = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_mat)
        return angles[0]  # pitch
