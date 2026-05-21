#!/usr/bin/env python3
"""
HSHL Line Following Student Lab — Your Implementation
=====================================================

Implement your line following algorithm by filling in the function below:

    detect_line(image)  — called for every camera frame (~30 fps)

─────────────────────────────────────────────────────────────────────────────
INPUTS  (what you receive from the camera)
─────────────────────────────────────────────────────────────────────────────
Camera frame  →  detect_line(image)
  image         np.ndarray, shape (720, 1280, 3), BGR colour order
                Same convention as OpenCV.

─────────────────────────────────────────────────────────────────────────────
OUTPUTS  (what your function must return)
─────────────────────────────────────────────────────────────────────────────
detect_line(image)  →  float | None
  Return a steering value in range [-1.0, 1.0]:
    -1.0  = steer full left
     0.0  = go straight (line is centered)
    +1.0  = steer full right
        None  = cannot detect line (framework uses neutral steering fallback)

─────────────────────────────────────────────────────────────────────────────
ALGORITHM TIPS
─────────────────────────────────────────────────────────────────────────────
1. The line is painted GREEN on the road (BGR: 0, 255, 0)
2. Use color range thresholding to detect green pixels
3. Find the line center using contour moments
4. Compare line center to image center to get steering offset
5. Use morphological operations to reduce noise
6. Return None if no line is detected

See docs/line_detection_example.py for a complete example implementation.

─────────────────────────────────────────────────────────────────────────────
HELPERS
─────────────────────────────────────────────────────────────────────────────
    self.show_notification(text)  white  — general info
    self.show_warning(text)       yellow — caution
    self.show_alert(text)         red    — critical
    self.current_image            latest camera frame (or None)
"""
#!/usr/bin/env python3
import cv2
import numpy as np
import rclpy
import joblib
import os
from collections import deque

from .interface import LineFollowingInterface


class MyLineFollower(LineFollowingInterface):

    def __init__(self):
        super().__init__("my_line_follower")

        # ── PID gains ────────────────────────────────────────────────────────
        self._Kp = 0.25   # proportional — how hard to correct
        self._Ki = 0.003  # integral     — corrects long-term drift
        self._Kd = 0.18   # derivative   — dampens overshoot on curves

        # ── PID state ────────────────────────────────────────────────────────
        self._prev_error = 0.0
        self._integral   = 0.0

        # ── Smoothing ────────────────────────────────────────────────────────
        # EMA on final steer output
        self._alpha      = 0.28
        self._prev_steer = 0.0

        # Rolling median filter on offset (5 frames) — kills single-frame noise
        self._offset_buffer = deque(maxlen=5)

        # ── Load SVM ─────────────────────────────────────────────────────────
        self.svm = joblib.load("team5_svm_final.pkl")
        self.get_logger().info("SVM loaded.")

        self.on_camera_image(self.detect_line)

    # ─────────────────────────────────────────────────────────────────────────
    def _get_offset(self, image: np.ndarray):
        """
        Returns normalised lateral offset of the green line centre.
        Negative = line is LEFT of image centre → steer left.
        Returns None if no line found.
        """
        h, w = image.shape[:2]

        # Bottom 55% only — ignore sky/buildings
        roi = image[int(h * 0.45):, :]
        rh, rw = roi.shape[:2]

        # HSV green mask
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([35, 50, 50]),
                           np.array([85, 255, 255]))

        # Denoise
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

        # Need a meaningful contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 150:
            return None

        # Use bottom 30% of ROI for immediate ahead — more reactive on curves
        strip = mask[int(0.70 * rh):, :]
        if cv2.countNonZero(strip) < 20:
            # Fall back to full contour centroid
            M = cv2.moments(largest)
            if M["m00"] == 0:
                return None
            cx = M["m10"] / M["m00"]
        else:
            _, xx = np.mgrid[0:strip.shape[0], 0:rw]
            cx = float((strip * xx).sum() / (strip.sum() + 1e-8))

        offset = (cx - rw / 2.0) / (rw / 2.0)
        return float(np.clip(offset, -1.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    def detect_line(self, image: np.ndarray) -> float | None:

        offset = self._get_offset(image)

        # ── Lane lost ────────────────────────────────────────────────────────
        if offset is None:
            self.show_warning("No line")
            self._prev_steer *= 0.80   # gentle decay, don't snap to 0
            return float(self._prev_steer)

        # ── Rolling median filter on offset ──────────────────────────────────
        self._offset_buffer.append(offset)
        smooth_offset = float(np.median(self._offset_buffer))

        # ── SVM gates the controller ─────────────────────────────────────────
        prediction = int(self.svm.predict([[smooth_offset]])[0])

        if prediction == 0:
            # Centred — bleed off integral and steer gently to zero
            self._integral   *= 0.4
            self._prev_error  = 0.0
            raw_steer         = 0.0
        else:
            # PID on the smoothed offset
            self._integral   += smooth_offset
            self._integral    = float(np.clip(self._integral, -4.0, 4.0))  # anti-windup
            derivative        = smooth_offset - self._prev_error
            self._prev_error  = smooth_offset

            raw_steer = float(np.clip(
                self._Kp * smooth_offset +
                self._Ki * self._integral +
                self._Kd * derivative,
                -1.0, 1.0
            ))

        # ── EMA on final steer — smooths out curve entry ──────────────────
        final = self._alpha * raw_steer + (1.0 - self._alpha) * self._prev_steer
        self._prev_steer = final

        self.show_notification(
            f"offset={smooth_offset:+.2f}  pred={prediction}  steer={final:+.3f}"
        )
        return float(np.clip(final, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    follower = MyLineFollower()
    try:
        rclpy.spin(follower)
    except KeyboardInterrupt:
        pass
    finally:
        follower.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()