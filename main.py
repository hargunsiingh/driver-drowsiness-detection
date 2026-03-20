"""
Driver Drowsiness Detection System
Real-time webcam application that monitors driver alertness using
eye closure, yawning, and head pose analysis.
"""

import cv2
import numpy as np

from detector import DrowsinessDetector, LEFT_EYE, RIGHT_EYE, MOUTH_HORIZONTAL, MOUTH_VERTICAL
from alert import AlertManager


# ── UI Constants ──────────────────────────────────────────────────────────────

COLOR_GREEN = (72, 199, 142)       # Awake
COLOR_YELLOW = (0, 215, 255)       # Warning
COLOR_RED = (60, 60, 230)          # Drowsy
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_DARK_BG = (30, 30, 30)
COLOR_MESH = (100, 200, 100)

BORDER_THICKNESS = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = cv2.FONT_HERSHEY_PLAIN

# Face mesh connections for drawing (subset for clean look)
FACE_OVAL_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10
]


def draw_overlay(frame: np.ndarray, result, fps: float) -> np.ndarray:
    """Draw the UI overlay on the frame."""
    h, w, _ = frame.shape

    # ── Determine border color based on alert level ───────────────────
    if result.alert_level == 2:
        border_color = COLOR_RED
        status_text = "DROWSY!"
        status_color = COLOR_RED
    elif result.alert_level == 1:
        border_color = COLOR_YELLOW
        status_text = "WARNING"
        status_color = COLOR_YELLOW
    else:
        border_color = COLOR_GREEN
        status_text = "AWAKE"
        status_color = COLOR_GREEN

    # ── Draw border ───────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, h), border_color, BORDER_THICKNESS)

    # ── Top status bar ────────────────────────────────────────────────
    bar_height = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), COLOR_DARK_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Status text
    cv2.putText(
        frame, status_text, (15, 35),
        FONT, 0.9, status_color, 2, cv2.LINE_AA,
    )

    # Metrics
    if result.face_detected:
        metrics = f"EAR: {result.ear:.2f}  |  MAR: {result.mar:.2f}  |  Tilt: {result.head_tilt:.1f}deg"
    else:
        metrics = "No face detected"
    cv2.putText(
        frame, metrics, (180, 35),
        FONT, 0.55, COLOR_WHITE, 1, cv2.LINE_AA,
    )

    # FPS
    cv2.putText(
        frame, f"FPS: {fps:.0f}", (w - 100, 35),
        FONT, 0.55, COLOR_WHITE, 1, cv2.LINE_AA,
    )

    # ── Draw face mesh wireframe ──────────────────────────────────────
    if result.face_detected and result.landmarks:
        lm = result.landmarks

        # Face oval
        for i in range(len(FACE_OVAL_INDICES) - 1):
            pt1 = (int(lm[FACE_OVAL_INDICES[i]][0]), int(lm[FACE_OVAL_INDICES[i]][1]))
            pt2 = (int(lm[FACE_OVAL_INDICES[i + 1]][0]), int(lm[FACE_OVAL_INDICES[i + 1]][1]))
            cv2.line(frame, pt1, pt2, COLOR_MESH, 1, cv2.LINE_AA)

        # Draw eye outlines
        eye_color = COLOR_RED if result.eyes_closed else COLOR_GREEN
        for eye_indices in [LEFT_EYE, RIGHT_EYE]:
            pts = np.array(
                [(int(lm[i][0]), int(lm[i][1])) for i in eye_indices],
                dtype=np.int32,
            )
            cv2.polylines(frame, [pts], True, eye_color, 2, cv2.LINE_AA)

        # Draw mouth outline
        mouth_color = COLOR_YELLOW if result.yawning else COLOR_GREEN
        mouth_pts = []
        for top_idx, bot_idx in MOUTH_VERTICAL:
            mouth_pts.append((int(lm[top_idx][0]), int(lm[top_idx][1])))
            mouth_pts.append((int(lm[bot_idx][0]), int(lm[bot_idx][1])))
        mouth_pts.append((int(lm[MOUTH_HORIZONTAL[0]][0]), int(lm[MOUTH_HORIZONTAL[0]][1])))
        mouth_pts.append((int(lm[MOUTH_HORIZONTAL[1]][0]), int(lm[MOUTH_HORIZONTAL[1]][1])))
        for pt in mouth_pts:
            cv2.circle(frame, pt, 3, mouth_color, -1, cv2.LINE_AA)

    # ── Drowsy alert overlay ──────────────────────────────────────────
    if result.drowsy:
        # Flashing red overlay
        alert_overlay = frame.copy()
        cv2.rectangle(alert_overlay, (0, 0), (w, h), COLOR_RED, -1)
        cv2.addWeighted(alert_overlay, 0.15, frame, 0.85, 0, frame)

        # Large warning text
        text = "!! DROWSY - WAKE UP !!"
        text_size = cv2.getTextSize(text, FONT, 1.2, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - 60

        # Text background
        cv2.rectangle(
            frame,
            (text_x - 15, text_y - text_size[1] - 15),
            (text_x + text_size[0] + 15, text_y + 15),
            COLOR_RED, -1,
        )
        cv2.putText(
            frame, text, (text_x, text_y),
            FONT, 1.2, COLOR_WHITE, 3, cv2.LINE_AA,
        )

    # ── Individual indicator badges ───────────────────────────────────
    if result.face_detected:
        badge_y = bar_height + 20

        if result.eyes_closed:
            cv2.putText(frame, "EYES CLOSED", (15, badge_y + 20), FONT, 0.6, COLOR_RED, 2, cv2.LINE_AA)

        if result.yawning:
            cv2.putText(frame, "YAWNING", (15, badge_y + 50), FONT, 0.6, COLOR_YELLOW, 2, cv2.LINE_AA)

        if result.head_drooping:
            cv2.putText(frame, "HEAD DROOPING", (15, badge_y + 80), FONT, 0.6, COLOR_YELLOW, 2, cv2.LINE_AA)

    # ── Bottom info bar ───────────────────────────────────────────────
    info_text = "Press 'q' to quit  |  Driver Drowsiness Detection System"
    cv2.putText(
        frame, info_text, (15, h - 15),
        FONT_SMALL, 1.0, (150, 150, 150), 1, cv2.LINE_AA,
    )

    return frame


def main():
    """Main application loop."""
    print("=" * 55)
    print("   DRIVER DROWSINESS DETECTION SYSTEM")
    print("=" * 55)
    print("  Starting webcam... Press 'q' to quit.")
    print()

    # Initialise components
    detector = DrowsinessDetector()
    alert_mgr = AlertManager()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam. Check your camera connection.")
        return

    # Try to set webcam resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps = 0.0
    prev_tick = cv2.getTickCount()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                break

            # Mirror the frame for natural interaction
            frame = cv2.flip(frame, 1)

            # Run detection
            result = detector.process_frame(frame)

            # Calculate FPS
            curr_tick = cv2.getTickCount()
            time_elapsed = (curr_tick - prev_tick) / cv2.getTickFrequency()
            if time_elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / time_elapsed)  # smoothed FPS
            prev_tick = curr_tick

            # Manage alarm
            if result.drowsy:
                alert_mgr.trigger_alarm()
            else:
                alert_mgr.stop_alarm()

            # Draw UI
            frame = draw_overlay(frame, result, fps)

            # Display
            cv2.imshow("Driver Drowsiness Detector", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        alert_mgr.cleanup()
        detector.release()
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
