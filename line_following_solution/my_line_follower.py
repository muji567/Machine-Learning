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
import cv2          # type: ignore
import numpy as np  # type: ignore
import rclpy        # type: ignore
import joblib       # type: ignore
from collections import deque
import os

from .interface import LineFollowingInterface


class MyLineFollower(LineFollowingInterface):
    """
    Student implementation of line following using trained SVM.
    """

    def __init__(self):
        super().__init__("my_line_follower")
        self._frame_count = 0
        
        self._Kp = 0.42      
        self._Ki = 0.003      
        self._Kd = 0.45       
        
        self._prev_error = 0.0
        self._integral   = 0.0
        
        self._alpha      = 0.45
        self._prev_steer = 0.0
        
        self._offset_buffer = deque(maxlen=5)

        # Load trained SVM model
        model_path = os.path.join("team5_svm_final.pkl")
        self.svm = joblib.load(model_path)
        self.get_logger().info("SVM model loaded successfully.")
        
        self.on_camera_image(self.detect_line)
        self.get_logger().info("MyLineFollower initialized — SVM ready")

    def detect_line(self, image: np.ndarray) -> float | None:
        self._frame_count += 1
        h, w = image.shape[:2]

        # 1. ROI & Color Filtering
        roi = image[int(h * 0.45):, :]
        rh, rw = roi.shape[:2]
        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))

        # Clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 2. Extract Line Center
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < 150:
            self._prev_steer *= 0.80  # Smooth steering decay fallback
            return float(self._prev_steer)

        largest_contour = max(contours, key=cv2.contourArea)
        lookahead_strip = mask[int(0.70 * rh):, :]
        
        if cv2.countNonZero(lookahead_strip) < 20:
            moments = cv2.moments(largest_contour)
            if moments["m00"] == 0: return float(self._prev_steer)
            cx = moments["m10"] / moments["m00"]
        else:
            _, xx = np.mgrid[0:lookahead_strip.shape[0], 0:rw]
            cx = float((lookahead_strip * xx).sum() / (lookahead_strip.sum() + 1e-8))

        # 3. Running Median Smoothing
        raw_offset = (cx - rw / 2.0) / (rw / 2.0)
        self._offset_buffer.append(raw_offset)
        smooth_offset = float(np.median(self._offset_buffer))

        # 4. Balanced 5-Class Model Prediction Gating
        prediction = int(self.svm.predict([[smooth_offset]])[0])

        # 5. Actuator Command Formulation (PID Loop)
        if prediction == 0:
            self._integral  *= 0.4
            self._prev_error = 0.0
            raw_steer        = 0.0
        else:
            self._integral += smooth_offset
            self._integral  = float(np.clip(self._integral, -4.0, 4.0))
            derivative      = smooth_offset - self._prev_error
            self._prev_error = smooth_offset

            base_pid_steer = (self._Kp * smooth_offset +
                              self._Ki * self._integral +
                              self._Kd * derivative)

            # Boost steering by 40% if model identifies a HARD turn (-2 or 2)
            if abs(prediction) == 2:
                base_pid_steer *= 1.40

            raw_steer = float(np.clip(base_pid_steer, -1.0, 1.0))

        # 6. Apply Temporal Smoothing (EMA)
        final_steer = self._alpha * raw_steer + (1.0 - self._alpha) * self._prev_steer
        self._prev_steer = final_steer

        # Telemetry Notifications
        if self._frame_count % 30 == 0:
            self.get_logger().info(f"RUNNING: steer={final_steer:+.2f} class={prediction} frame={self._frame_count}")
        self.show_notification(f"offset={smooth_offset:+.2f} class={prediction} steer={final_steer:+.2f}")
        
        return float(np.clip(final_steer, -1.0, 1.0))

def main(args=None):
    """Main entry point for the line follower node."""
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