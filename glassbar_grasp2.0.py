#!/home/erlin/anaconda3/envs/foundationpose/bin/python3
"""Glassbar grasp routine built on the refactored level2 actions."""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np
import rospy

from camera_reader import CameraReader
from env import create_env
from level2_action import (
    DetectionResult,
    ContactDetectionResult,
    DirectionDetectionResult,
    GraspPlan,
    detect_glassbar_direction,
    detect_object_pose_using_foundation_pose,
    monitor_contact_with_camera,
    plan_grasp_pose,
    execute_grasp_plan,
    adjust_to_vertical_and_lift,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PIPELINE_STATE: dict[str, object] = {}


def _cleanup_resources() -> None:
    robot = PIPELINE_STATE.get("robot")
    gripper = PIPELINE_STATE.get("gripper")
    camera = PIPELINE_STATE.get("camera")
    angle_camera = PIPELINE_STATE.get("angle_camera")
    contact_camera = PIPELINE_STATE.get("contact_camera")

    try:
        if angle_camera and getattr(angle_camera, "cap", None):
            angle_camera.cap.release()
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        if contact_camera and getattr(contact_camera, "cap", None):
            contact_camera.cap.release()
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        if camera and hasattr(camera, "release"):
            camera.release()
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        if robot:
            robot.stop()
            robot.disable_robot()
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        if gripper:
            gripper.disconnect()
    except Exception:  # pylint: disable=broad-except
        pass
    cv2.destroyAllWindows()


def _signal_handler(signum, frame):  # type: ignore[override]
    logger.info("收到信号 %s，准备退出", signum)
    _cleanup_resources()
    try:
        rospy.signal_shutdown("Interrupted")
    except Exception:  # pylint: disable=broad-except
        pass
    sys.exit(0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Glassbar grasp routine")
    parser.add_argument("--config", default="config.json", help="Config path for env creation")
    parser.add_argument("--target", default="red stirring rod", help="GraspLibrary target key")
    parser.add_argument("--mesh", default=os.path.join("FoundationPose", "mesh", "thin_cube.obj"), help="Mesh path")
    parser.add_argument("--mask-mode", choices=["qwen", "groundingdino"], default="qwen", help="Segmentation backend")
    parser.add_argument("--qwen-model-path", default=os.environ.get("QWEN_VL_MODEL_PATH"), help="Qwen model path/id")
    parser.add_argument("--detection-interval", type=int, default=15, help="Run pose detection every N frames")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum frames to attempt pose detection")
    parser.add_argument("--record-root", default="record_images_during_grasp", help="Directory to store run logs")
    parser.add_argument("--debug", action="store_true", help="Enable extra debug outputs and pauses")
    parser.add_argument("--pause-after-detection", action="store_true", help="Pause after initial pose detection")
    return parser.parse_args()


def _save_detection_artifacts(save_dir: str, frame_idx: int, detection) -> None:
    color_path = os.path.join(save_dir, f"color_frame_{frame_idx:06d}.png")
    mask_path = os.path.join(save_dir, f"mask_frame_{frame_idx:06d}.png")
    vis_path = os.path.join(save_dir, f"vis_frame_{frame_idx:06d}.png")
    cv2.imwrite(color_path, detection.color_bgr)
    cv2.imwrite(mask_path, detection.mask)
    if detection.visualization_bgr is not None:
        cv2.imwrite(vis_path, detection.visualization_bgr)


def _display_detection_debug(detection) -> None:
    cv2.imshow("mask", detection.mask)
    cv2.imshow("color", detection.color_bgr)
    if detection.visualization_bgr is not None:
        cv2.imshow("object 6D pose", detection.visualization_bgr)
    cv2.waitKey(1)


def _run_detection_loop(args, camera_main, mesh_path: str) -> DetectionResult:
    detection_result: Optional[DetectionResult] = None
    frame_count = 0
    max_frames = max(1, args.max_frames)

    while frame_count < max_frames:
        frames = camera_main["cam"].get_frames()
        if frames is None:
            continue

        should_detect = (frame_count % args.detection_interval) == 0 or detection_result is None
        if should_detect:
            detection_result = detect_object_pose_using_foundation_pose(
                args.target,
                mesh_path,
                camera_main,
                mask_mode=args.mask_mode,
                qwen_model_path=args.qwen_model_path,
                debug=args.debug,
                frames=frames,
            )
            _display_detection_debug(detection_result)
            if args.pause_after_detection:
                logger.info("按任意键继续，或在终端输入回车")
                cv2.waitKey(0)
                try:
                    input("继续？按回车…")
                except EOFError:
                    pass
            break
        frame_count += 1

    if detection_result is None:
        raise RuntimeError("未能在限定帧数内检测到目标物体位姿")
    return detection_result


def _execute_grasp_sequence(
    plan: GraspPlan,
    robot_main: Dict[str, Any],
    gripper,
) -> bool:
    logger.info("执行抓取动作…")
    return execute_grasp_plan(
        robot_main,
        gripper,
        plan,
        gripper_open_position=plan.parameters.get("gripper_open_pos", 1000),
        gripper_close_force=plan.parameters.get("gripper_force", 80),
        gripper_close_speed=plan.parameters.get("gripper_speed", 30),
    )


def _run_direction_adjustment(
    angle_camera: CameraReader,
    robot,
    grasp_plan: GraspPlan,
    save_dir: str,
) -> DirectionDetectionResult:
    direction_result = detect_glassbar_direction(angle_camera, save_dir=save_dir)
    avg_angle = abs(direction_result.avg_angle)
    grasp_tilt_angle = float(grasp_plan.parameters.get("grasp_tilt_angle", 0.0))
    adjust_info = adjust_to_vertical_and_lift(
        robot,
        avg_angle=avg_angle,
        grasp_tilt_angle=grasp_tilt_angle,
    )
    logger.info("姿态调整结果: %s", adjust_info)
    return direction_result


def _monitor_surface_contact(
    contact_camera: CameraReader,
    robot,
    save_dir: str,
) -> ContactDetectionResult:
    gray_debug_dir = os.path.join(save_dir, "gray_images_debug")
    os.makedirs(gray_debug_dir, exist_ok=True)

    def _debug_callback(before, after, step):  # noqa: ANN001
        cv2.imshow("contact_before", before)
        cv2.imshow("contact_after", after)
        cv2.waitKey(1)

    result = monitor_contact_with_camera(
        contact_camera,
        robot,
        sample_interval=0.1,
        move_step=3,
        max_steps=700,
        change_threshold=3,
        debug_dir=gray_debug_dir,
        on_debug=_debug_callback,
    )
    logger.info("接触检测结果: %s", result)
    return result


def main(args: argparse.Namespace) -> None:
    try:
        rospy.init_node("glassbar_grasp", anonymous=True, disable_signals=True)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("ROS节点初始化失败: %s", exc)

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    env = create_env(config_path)
    robot_main = env.robot1
    gripper = env.gripper
    camera_main = env.camera1_main

    PIPELINE_STATE.update({
        "robot": robot_main["robot"],
        "gripper": gripper,
        "camera": camera_main["cam"],
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_root = os.path.abspath(args.record_root)
    os.makedirs(record_root, exist_ok=True)
    session_dir = os.path.join(record_root, timestamp)
    os.makedirs(session_dir, exist_ok=True)

    logger.info("运行记录目录: %s", session_dir)

    detection = _run_detection_loop(args, camera_main, os.path.abspath(args.mesh))
    _save_detection_artifacts(session_dir, 0, detection)

    plan = plan_grasp_pose(args.target, detection.center_pose, camera_main, robot_main)
    success = _execute_grasp_sequence(plan, robot_main, gripper)

    #! 注意不能反了
    angle_camera = CameraReader(camera_id=10, init_camera=True)
    contact_camera = CameraReader(camera_id=11, init_camera=True)
    PIPELINE_STATE.update({
        "angle_camera": angle_camera,
        "contact_camera": contact_camera,
    })

    direction_result = _run_direction_adjustment(
        angle_camera,
        robot_main["robot"],
        plan,
        save_dir=session_dir,
    )

    contact_result = _monitor_surface_contact(contact_camera, robot_main["robot"], session_dir)

    report = {
        "target": args.target,
        "mesh": os.path.abspath(args.mesh),
        "success": success,
        "direction": asdict(direction_result),
        "contact": asdict(contact_result),
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(session_dir, "grasp_report.json"), "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)




if __name__ == "__main__":
    parsed_args = _parse_args()
    PIPELINE_STATE["args"] = parsed_args
    atexit.register(_cleanup_resources)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    main(parsed_args)


