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
import cv2          # type: ignore
import numpy as np  # type: ignore
import rclpy        # type: ignore
import joblib       # type: ignore
import os

from .interface import LineFollowingInterface


class MyLineFollower(LineFollowingInterface):
    """
    Student implementation of line following using trained SVM.
    """

    def __init__(self):
        super().__init__("my_line_follower")
        self._frame_count = 0

        self._Kp = 0.4
        self._Ki = 0.01
        self._Kd = 0.1
        self._prev_error = 0.0
        self._integral = 0.0
    
        # Load trained SVM model
        model_path = os.path.join("team5_svm_final.pkl")
        self.svm = joblib.load(model_path)
        self.get_logger().info("SVM model loaded successfully.")
    
        self.on_camera_image(self.detect_line)
        self.get_logger().info("MyLineFollower initialized — SVM ready")

    def detect_line(self, image: np.ndarray) -> float | None:
        """
        Detect the green line using HSV masking and steer using trained SVM.
        
        Args:
            image: BGR image from camera, shape (720, 1280, 3)
        
        Returns:
            Steering value in [-1.0, 1.0], or None if line not detected.
        """
        # Step 1 — Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Step 1b — Crop to ROI (bottom 55% of image)
        h, w = hsv.shape[:2]
        roi_start = int(h * 0.45)
        hsv = hsv[roi_start:, :]

        # Step 2 — Green color mask
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Step 3 — Reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Step 4 — Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < 100:
            self.show_warning("No line detected")
            return None

        largest_contour = max(contours, key=cv2.contourArea)

        # Step 5 & 6 — Calculate bottom_offset
        # (from Lab07 extract_extended_features)
        h, w = mask.shape
        bottom = mask[int(0.70*h):, :].astype(float)
        eps = 1e-8

        if bottom.sum() < eps:
            return None

        _, xx = np.mgrid[0:bottom.shape[0], 0:w]
        cx_bottom = (bottom * xx).sum() / (bottom.sum() + eps)
        offset = float((cx_bottom - w/2) / (w/2))
        offset = float(np.clip(offset, -1.0, 1.0))

        # Step 7 — Use SVM to predict steering class
        prediction = int(self.svm.predict([[offset]])[0])

        # Step 8 — PID Controller
        error = offset

        if prediction == 0:
            self._integral = 0.0
            self._prev_error = 0.0
            steering = 0.0
        else:
            self._integral += error
            derivative = error - self._prev_error
            self._prev_error = error

            steering = float(np.clip(
                self._Kp * error +
                self._Ki * self._integral +
                self._Kd * derivative,
                -1.0, 1.0
            ))

        self.show_notification(f"steer={steering:.2f} pred={prediction}")
        return steering


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
