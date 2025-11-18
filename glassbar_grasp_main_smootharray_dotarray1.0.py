#!/usr/bin/env python3
"""Glassbar grasp entry point adapted to the RoboTwin-style framework."""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np
import rospy

from env import create_env
from level2_action import (
    DetectionResult,
    GraspPlan,
    choose_grasp_pose,
    detect_object_pose_using_foundation_pose,
    execute_grasp_plan,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(ROOT_DIR, "config.json")
DEFAULT_RECORD_ROOT = os.path.join(ROOT_DIR, "record_images_during_grasp")
DEFAULT_TARGET = "red cylinder"
DEFAULT_MESH_PATH = os.path.join(ROOT_DIR, "FoundationPose", "mesh", "thin_cube.obj")
DEFAULT_QWEN_PATH = os.environ.get(
    "QWEN_VL_MODEL_PATH",
    os.path.join(ROOT_DIR, "Qwen3-VL", "Qwen3-VL-4B-Thinking"),
)

PIPELINE: Optional["GlassbarGraspPipeline"] = None


@dataclass
class RunSummary:
    target: str
    mesh_path: str
    detection_frame: int
    detection: Dict[str, Any]
    grasp_plan: Dict[str, Any]
    success: bool
    timestamp: str


class GlassbarGraspPipeline:
    def __init__(
        self,
        *,
        config_path: str,
        target: str,
        mesh_path: str,
        mask_mode: str,
        qwen_model_path: Optional[str],
        detection_interval: int,
        record_root: str,
        max_frames: int,
        debug: bool,
    ):
        self.target = target
        self.mesh_path = mesh_path
        self.mask_mode = mask_mode
        self.qwen_model_path = qwen_model_path
        self.detection_interval = max(1, detection_interval)
        self.max_frames = max(1, max_frames)
        self.debug = debug

        logging.info("Loading environment with config %s", config_path)
        self.env = create_env(config_path)
        self.robot_main = self.env.robot1
        self.gripper = self.env.gripper
        self.camera_main = self.env.camera1_main

        os.makedirs(record_root, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(record_root, timestamp)
        os.makedirs(self.session_dir, exist_ok=True)
        self.debug_dir = os.path.join(self.session_dir, "debug") if debug else None
        if self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)

        self._cleaned = False

    def run(self, *, dry_run: bool = False) -> RunSummary:
        logging.info("Starting detection for target '%s'", self.target)
        detection, frame_idx = self._detect_target_pose()
        plan = choose_grasp_pose(
            target=self.target,
            center_pose=detection.center_pose,
            cam_main=self.camera_main,
            robot_main=self.robot_main,
        )
        logging.info("Computed grasp plan for target '%s'", self.target)

        success = True
        if not dry_run:
            logging.info("Executing grasp plan...")
            success = execute_grasp_plan(
                robot_main=self.robot_main,
                gripper=self.gripper,
                grasp_plan=plan,
                gripper_open_position=plan.parameters.get("gripper_open_pos", 1000),
            )
            logging.info("Grasp execution %s", "succeeded" if success else "failed")
        else:
            logging.info("Dry-run mode: skipping grasp execution")

        summary = RunSummary(
            target=self.target,
            mesh_path=self.mesh_path,
            detection_frame=frame_idx,
            detection={
                "center_pose": detection.center_pose.tolist(),
                "raw_pose": detection.raw_pose.tolist(),
                "to_origin": detection.to_origin.tolist(),
            },
            grasp_plan={
                "pre_grasp_pose": list(plan.pre_grasp_pose),
                "grasp_pose": list(plan.grasp_pose),
                "z_safe_distance": plan.z_safe_distance,
                "gripper_close_pos": plan.gripper_close_pos,
                "parameters": plan.parameters,
            },
            success=success,
            timestamp=datetime.now().isoformat(),
        )
        self._persist_results(detection, plan, summary)
        return summary

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        logging.info("Cleaning up resources")
        try:
            cv2.destroyAllWindows()
        except Exception:  # pylint: disable=broad-except
            pass
        try:
            if self.camera_main and self.camera_main.get("cam"):
                self.camera_main["cam"].release()
        except Exception as exc:  # pylint: disable=broad-except
            logging.debug("Failed to release camera: %s", exc)
        try:
            if self.gripper:
                self.gripper.disconnect()
        except Exception as exc:  # pylint: disable=broad-except
            logging.debug("Failed to disconnect gripper: %s", exc)
        try:
            if self.robot_main and self.robot_main.get("robot"):
                robot = self.robot_main["robot"]
                robot.stop()
                robot.disable_robot()
        except Exception as exc:  # pylint: disable=broad-except
            logging.debug("Failed to stop robot: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _detect_target_pose(self) -> tuple[DetectionResult, int]:
        camera = self.camera_main["cam"]
        detection: Optional[DetectionResult] = None
        mask_kwargs: Dict[str, Any] = {}
        for frame_idx in range(self.max_frames):
            frames = camera.get_frames()
            if frames is None:
                continue
            color = frames["color"]
            self._save_frame(color, frame_idx)

            should_detect = (frame_idx % self.detection_interval) == 0 or detection is None
            if not should_detect:
                continue

            if self.debug_dir:
                mask_kwargs["bbox_vis_path"] = os.path.join(
                    self.debug_dir, f"bbox_{frame_idx:06d}.png"
                )
            try:
                detection = detect_object_pose_using_foundation_pose(
                    target=self.target,
                    mesh_path=self.mesh_path,
                    cam_main=self.camera_main,
                    mask_mode=self.mask_mode,
                    qwen_model_path=self.qwen_model_path,
                    mask_kwargs=mask_kwargs,
                    frames=frames,
                    debug=self.debug,
                    debug_dir=self.debug_dir,
                )
                self._save_detection_debug(detection, frame_idx)
                logging.info("Detection succeeded on frame %d", frame_idx)
                return detection, frame_idx
            except Exception as exc:  # pylint: disable=broad-except
                logging.exception("Detection failed on frame %d: %s", frame_idx, exc)
                time.sleep(0.05)

        raise RuntimeError("Failed to detect target pose within the allotted frames")

    def _save_frame(self, color: np.ndarray, frame_idx: int) -> None:
        color_path = os.path.join(self.session_dir, f"color_{frame_idx:06d}.png")
        cv2.imwrite(color_path, color)

    def _save_detection_debug(self, detection: DetectionResult, frame_idx: int) -> None:
        mask_path = os.path.join(self.session_dir, f"mask_{frame_idx:06d}.png")
        cv2.imwrite(mask_path, detection.mask)
        if detection.debug_visual_bgr is not None:
            vis_path = os.path.join(self.session_dir, f"vis_{frame_idx:06d}.png")
            cv2.imwrite(vis_path, detection.debug_visual_bgr)

    def _persist_results(
        self,
        detection: DetectionResult,
        plan: GraspPlan,
        summary: RunSummary,
    ) -> None:
        np.save(
            os.path.join(self.session_dir, "center_pose.npy"),
            detection.center_pose,
        )
        report_path = os.path.join(self.session_dir, "grasp_report.json")
        with open(report_path, "w", encoding="utf-8") as file:
            json.dump(asdict(summary), file, indent=2, ensure_ascii=False)


def _handle_signal(signum, frame):  # type: ignore[override]
    logging.info("Received signal %s, shutting down", signum)
    if PIPELINE is not None:
        PIPELINE.cleanup()
    sys.exit(0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Glassbar grasp pipeline")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.json")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Object name in GraspLibrary.json")
    parser.add_argument("--mesh", default=DEFAULT_MESH_PATH, help="Mesh path used by FoundationPose")
    parser.add_argument(
        "--mask-mode",
        default="qwen",
        choices=["qwen", "groundingdino"],
        help="Segmentation backend",
    )
    parser.add_argument(
        "--qwen-model-path",
        default=DEFAULT_QWEN_PATH,
        help="Path or model id for Qwen3-VL (used when mask-mode=qwen)",
    )
    parser.add_argument(
        "--detection-interval",
        type=int,
        default=15,
        help="Run pose detection every N frames",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="Maximum number of frames to search for the target",
    )
    parser.add_argument(
        "--record-root",
        default=DEFAULT_RECORD_ROOT,
        help="Directory to store session logs",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug outputs")
    parser.add_argument("--dry-run", action="store_true", help="Skip actual robot execution")
    return parser.parse_args()


def main() -> None:
    global PIPELINE  # pylint: disable=global-statement

    args = _parse_args()

    try:
        rospy.init_node("glassbar_grasp", anonymous=True, disable_signals=True)
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("Failed to initialise ROS node: %s", exc)

    PIPELINE = GlassbarGraspPipeline(
        config_path=args.config,
        target=args.target,
        mesh_path=args.mesh,
        mask_mode=args.mask_mode,
        qwen_model_path=args.qwen_model_path,
        detection_interval=args.detection_interval,
        record_root=args.record_root,
        max_frames=args.max_frames,
        debug=args.debug,
    )

    atexit.register(lambda: PIPELINE and PIPELINE.cleanup())
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    summary = PIPELINE.run(dry_run=args.dry_run)
    logging.info("Run summary saved to %s", os.path.join(PIPELINE.session_dir, "grasp_report.json"))
    logging.info("Outcome: %s", "success" if summary.success else "failure")


if __name__ == "__main__":
    main()


