#!/usr/bin/env python3
"""Simple ROS-based monitor to verify pixel change detection logic.

The script subscribes to the `raw_image` topic, grabs consecutive frames at a
fixed interval (default 0.1s), and reports whether the pixel change detector
from `camera_reader.CameraReader` flags a significant difference.

Usage (run within the ROS workspace environment):

    python3 test_pixel_change_monitor.py --change-threshold 1.0 --debug

Press Ctrl+C to exit.
"""

import argparse
import os
import threading
from datetime import datetime

import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from camera_reader import CameraReader


class PixelChangeMonitor:
    """Continuously compares consecutive frames to verify change detection."""

    def __init__(
        self,
        topic_name: str,
        compare_interval: float,
        change_threshold: float,
        pixel_threshold: int,
        min_area: int,
        save_debug: bool,
    ) -> None:
        self.topic_name = topic_name
        self.compare_interval = compare_interval
        self.change_threshold = change_threshold
        self.pixel_threshold = pixel_threshold
        self.min_area = min_area
        self.save_debug = save_debug

        self.bridge = CvBridge()
        self.detector = CameraReader(init_camera=False)

        self._latest_frame = None
        self._latest_stamp = None
        self._frame_lock = threading.Lock()

        self._prev_frame = None
        self._prev_stamp = None
        self._compare_count = 0

        if self.save_debug:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_dir = os.path.join(
                "record_images_during_grasp",
                timestamp,
                "pixel_change_monitor",
            )
            os.makedirs(self.debug_dir, exist_ok=True)
            rospy.loginfo("[PixelChangeMonitor] Saving debug frames to %s", self.debug_dir)
        else:
            self.debug_dir = None

        rospy.Subscriber(self.topic_name, Image, self._image_callback, queue_size=1)
        rospy.loginfo("[PixelChangeMonitor] Subscribed to %s", self.topic_name)

    # ------------------------------------------------------------------
    def _image_callback(self, msg: Image) -> None:
        """Receive raw images and cache the most recent frame."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as err:  # pylint: disable=broad-except
            rospy.logerr("Failed to convert image message: %s", err)
            return

        with self._frame_lock:
            self._latest_frame = frame.copy()
            self._latest_stamp = msg.header.stamp.to_sec() if msg.header.stamp else rospy.get_time()

    # ------------------------------------------------------------------
    def _fetch_latest_frame(self):
        with self._frame_lock:
            if self._latest_frame is None:
                return None, None
            return self._latest_frame.copy(), self._latest_stamp

    # ------------------------------------------------------------------
    def spin(self) -> None:
        rate = rospy.Rate(1.0 / self.compare_interval)
        rospy.loginfo(
            "[PixelChangeMonitor] Monitoring pixel changes every %.3f seconds", self.compare_interval
        )

        while not rospy.is_shutdown():
            frame, stamp = self._fetch_latest_frame()

            if frame is None or stamp is None:
                rospy.logwarn_throttle(5.0, "[PixelChangeMonitor] Waiting for frames...")
                rate.sleep()
                continue

            if self._prev_frame is None:
                self._prev_frame = frame
                self._prev_stamp = stamp
                rate.sleep()
                continue

            if (stamp - self._prev_stamp) < self.compare_interval * 0.9:
                rate.sleep()
                continue

            compare_idx = self._compare_count
            result = self.detector.detect_pixel_changes(
                self._prev_frame,
                frame,
                threshold=self.pixel_threshold,
                min_area=self.min_area,
                save_dir=self.debug_dir,
                step_num=compare_idx if self.debug_dir else None,
            )

            if "error" in result:
                rospy.logwarn("[PixelChangeMonitor] Detection error: %s", result["error"])
            else:
                change_pct = result["change_percentage"]
                has_change = change_pct >= self.change_threshold
                rospy.loginfo(
                    "[PixelChangeMonitor] #%d Δt=%.3fs change=%.4f%% => %s",
                    compare_idx,
                    stamp - self._prev_stamp,
                    change_pct,
                    "CHANGE" if has_change else "stable",
                )

            self._prev_frame = frame
            self._prev_stamp = stamp
            self._compare_count += 1

            rate.sleep()


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor consecutive frames for pixel changes.")
    parser.add_argument(
        "--topic",
        default="/raw_image",
        help="Image topic to subscribe to (default: /raw_image)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Comparison interval in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=0.1,
        help="Percentage threshold to flag a change",
    )
    parser.add_argument(
        "--pixel-threshold",
        type=int,
        default=3,
        help="Pixel intensity difference threshold",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=2,
        help="Minimum contour area to count as change",
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable saving debug gray/diff images (enabled by default).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    rospy.init_node("pixel_change_monitor", anonymous=True)

    monitor = PixelChangeMonitor(
        topic_name=args.topic,
        compare_interval=args.interval,
        change_threshold=args.change_threshold,
        pixel_threshold=args.pixel_threshold,
        min_area=args.min_area,
        save_debug=not args.no_debug,
    )

    try:
        monitor.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()

