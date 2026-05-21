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
        self._Kp = 0.42   # Proportional - slightly relaxed to prevent initial aggressive snaps
        self._Ki = 0.003  # Integral - corrects long-term systematic drift
        self._Kd = 0.45   # Derivative - aggressively dampens overshoot oscillations on bends

        # ── PID state ────────────────────────────────────────────────────────
        self._prev_error = 0.0
        self._integral   = 0.0

        # ── Smoothing & Phase Lag Fix ────────────────────────────────────────
        # EMA on final steer output. Shifted from 0.28 to 0.45 to reduce phase lag
        # on sudden turn transitions while allowing the median filter to handle noise.
        self._alpha      = 0.45
        self._prev_steer = 0.0

        # Rolling median filter on offset (5 frames) — filters out camera dropouts
        self._offset_buffer = deque(maxlen=5)

        # ── Load Upgraded Balanced 5-Class SVM ───────────────────────────────
        self.svm = joblib.load("team5_svm_5class_balanced.pkl")
        self.get_logger().info("Production 5-Class Balanced SVM model loaded successfully.")

        self.on_camera_image(self.detect_line)

    # ─────────────────────────────────────────────────────────────────────────
    def _get_offset(self, image: np.ndarray):
        """
        Returns normalized lateral offset of the green line centre.
        Negative = line is LEFT of image centre → steer left.
        Returns None if no line found.
        """
        h, w = image.shape[:2]

        # Extract Region of Interest (ROI) - Bottom 55% only to eliminate background noise
        roi = image[int(h * 0.45):, :]
        rh, rw = roi.shape[:2]

        # Apply HSV green mask matching track guidelines
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([35, 50, 50]),
                           np.array([85, 255, 255]))

        # Morphological Filtering to reduce salt-and-pepper sensor noise
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

        # Ensure structured contour elements exist within frame bounds
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 150:
            return None

        # Lookahead Slicing: Target bottom 30% of ROI for responsive immediate feedback
        strip = mask[int(0.70 * rh):, :]
        if cv2.countNonZero(strip) < 20:
            # Fall back to global contour moments if immediate patch is lost
            M = cv2.moments(largest)
            if M["m00"] == 0:
                return None
            cx = M["m10"] / M["m00"]
        else:
            # Vectorized center calculation inside the localized search strip
            _, xx = np.mgrid[0:strip.shape[0], 0:rw]
            cx = float((strip * xx).sum() / (strip.sum() + 1e-8))

        offset = (cx - rw / 2.0) / (rw / 2.0)
        return float(np.clip(offset, -1.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    def detect_line(self, image: np.ndarray) -> float | None:

        offset = self._get_offset(image)

        # ── Lane lost fallback mechanism ─────────────────────────────────────
        if offset is None:
            self.show_warning("Line Lost - Activating steering decay tracking")
            self._prev_steer *= 0.80   # Gentle decay curve to maintain arc without sudden snapping
            return float(self._prev_steer)

        # ── Rolling median filter tracking ───────────────────────────────────
        self._offset_buffer.append(offset)
        smooth_offset = float(np.median(self._offset_buffer))

        # ── Balanced 5-Class SVM Decision Logic ──────────────────────────────
        # Model returns classification: -2 (HARD L), -1 (SOFT L), 0 (STR), 1 (SOFT R), 2 (HARD R)
        prediction = int(self.svm.predict([[smooth_offset]])[0])

        if prediction == 0:
            # STR: Bleed off integral wind-up and guide smoothly back toward central axis
            self._integral   *= 0.4
            self._prev_error  = 0.0
            raw_steer         = 0.0
        else:
            # Curves: Compute full parallel PID loop on the smoothed spatial tracking coordinate
            self._integral   += smooth_offset
            self._integral    = float(np.clip(self._integral, -4.0, 4.0))  # Anti-windup clamping
            derivative        = smooth_offset - self._prev_error
            self._prev_error  = smooth_offset

            # Assemble core PID command
            base_pid_steer = (self._Kp * smooth_offset +
                              self._Ki * self._integral +
                              self._Kd * derivative)

            # Dynamic Compensation Gate: If the AI identifies a HARD turning profile (2 or -2),
            # dynamically scale up steering authority to combat centrifugal forces on corners.
            if abs(prediction) == 2:
                base_pid_steer *= 1.40

            raw_steer = float(np.clip(base_pid_steer, -1.0, 1.0))

        # ── Exponential Moving Average (EMA) Temporal Filter ────────────────
        final = self._alpha * raw_steer + (1.0 - self._alpha) * self._prev_steer
        self._prev_steer = final

        self.show_notification(
            f"offset={smooth_offset:+.2f}  class_pred={prediction}  steer_out={final:+.3f}"
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