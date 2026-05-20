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

from .interface import LineFollowingInterface


class MyLineFollower(LineFollowingInterface):
    """
    Student implementation of line following.
    
    Detect a green line and steer to stay centered on it.
    """

    def __init__(self):
        super().__init__("my_line_follower")
        self._frame_count = 0
        
        # Register camera callback
        self.on_camera_image(self.detect_line)
        self.get_logger().info("MyLineFollower initialized — ready to detect green line")

    def detect_line(self, image: np.ndarray) -> float | None:
        """
        Detect the green line and return steering command.
        
        Args:
            image: BGR image from camera, shape (720, 1280, 3)
        
        Returns:
            Steering value in [-1.0, 1.0], or None if line not detected.
        """
        # Convert to HSV (more robust to lighting)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Green in HSV: H ≈ 60-90 (roughly)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Dilate to connect nearby points
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < 100:
            self.show_warning("No line detected")
            return None

        largest_contour = max(contours, key=cv2.contourArea)

        # Fit a line to the contour
        rows, cols = mask.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)

        center_row = rows // 2
        if abs(vy) > 0.01:
            line_x_at_center = x + (center_row - y) * (vx / vy)
        else:
            M = cv2.moments(largest_contour)
            line_x_at_center = M["m10"] / M["m00"] if M["m00"] > 0 else cols // 2

        # Calculate steering
        image_center_x = cols / 2.0
        offset = float(np.squeeze(line_x_at_center - image_center_x) / image_center_x)
        steering = float(np.clip(offset * 0.5, -1.0, 1.0))

        self.show_notification(f"steer={steering:.2f}")
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
