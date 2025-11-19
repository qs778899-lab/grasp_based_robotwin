"""
Author: zhangcongshe

Date: 2025/11/1

Version: 1.0
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import rospy
import trimesh
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3, SO3

# Add FoundationPose to path for internal imports
_fp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FoundationPose")
if _fp_dir not in sys.path:
    sys.path.insert(0, _fp_dir)

import nvdiffrast.torch as dr
from estimater import FoundationPose
from learning.training.predict_pose_refine import PoseRefinePredictor
from learning.training.predict_score import ScorePredictor

from Utils import (
    draw_posed_3d_box,
    draw_xyz_axis,
    set_logging_format,
    set_seed,
)
from dino_mask import get_mask_from_GD
from qwen_mask import get_mask_from_qwen

default_logger = logging.getLogger(__name__)

from grasp_utils import (
    detect_dent_orientation,
    generate_spiral_offsets,
    measure_max_force,
    normalize_angle,
    print_pose_info,
)

GRASP_LIBRARY_PATH = os.path.join(os.path.dirname(__file__), "GraspLibrary.json")
_GRASP_LIBRARY_CACHE: Optional[Tuple[str, Dict[str, Any]]] = None


@dataclass
class DetectionResult:
    target: str
    center_pose: np.ndarray
    raw_pose: np.ndarray
    to_origin: np.ndarray
    bbox: np.ndarray
    mask: np.ndarray
    color_bgr: np.ndarray
    depth_m: np.ndarray
    visualization_bgr: Optional[np.ndarray] = None


@dataclass
class GraspPlan:
    target: str
    pre_grasp_pose: Tuple[float, float, float, float, float, float]
    grasp_pose: Tuple[float, float, float, float, float, float]
    z_safe_distance: float
    gripper_close_pos: int
    T_base_ee_ideal: SE3
    parameters: Dict[str, Any]


@dataclass
class DirectionDetectionResult:
    angles: Iterable[float]
    avg_angle: float
    attempts: int
    timestamp: float


@dataclass
class ContactDetectionResult:
    contact_detected: bool
    steps_taken: int
    descent_distance: float
    final_pose: Iterable[float]


@dataclass
class ForceInsertionResult:
    success: bool
    total_descent: float
    planar_attempts: int
    planar_successes: int
    contact_events: Iterable[Dict[str, float]]
    cumulative_spiral_points: int
    final_pose: Iterable[float]


def _load_grasp_library(library_path: Optional[str] = None) -> Dict[str, Any]:
    global _GRASP_LIBRARY_CACHE
    resolved_path = os.path.abspath(library_path or GRASP_LIBRARY_PATH)
    if _GRASP_LIBRARY_CACHE and _GRASP_LIBRARY_CACHE[0] == resolved_path:
        return _GRASP_LIBRARY_CACHE[1]

    with open(resolved_path, "r", encoding="utf-8") as file:
        library = json.load(file)

    _GRASP_LIBRARY_CACHE = (resolved_path, library)
    return library


def _generate_mask(
    color_bgr: np.ndarray,
    target: str,
    mask_mode: str,
    qwen_model_path: Optional[str],
    mask_kwargs: Dict[str, Any],
) -> np.ndarray:
    mode = mask_mode.lower()
    if mode in {"qwen", "qwen3", "vl"}:
        mask = get_mask_from_qwen(color_bgr, target, model_path=qwen_model_path, **mask_kwargs)
    elif mode in {"groundingdino", "dino", "gd"}:
        mask = get_mask_from_GD(color_bgr, target, **mask_kwargs)
    else:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")

    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    return mask


def detect_object_pose_using_foundation_pose(
    target: str,
    mesh_path: str,
    cam_main: Dict[str, Any],
    *,
    mask_mode: str = "qwen",
    qwen_model_path: Optional[str] = None,
    mask_kwargs: Optional[Dict[str, Any]] = None,
    iteration: int = 50,
    debug: bool = False,
    debug_dir: Optional[str] = None,
    frames: Optional[Dict[str, np.ndarray]] = None,
    logger: logging.Logger = default_logger,
) -> DetectionResult:
    """Use FoundationPose and image masks to estimate object pose."""

    mask_kwargs = mask_kwargs or {}
    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(mesh_path, force="mesh")
    mesh = mesh.copy()
    mesh.vertices = (mesh.vertices / 1000.0).astype(np.float64)
    if mesh.vertex_normals is not None:
        mesh.vertex_normals = mesh.vertex_normals.astype(np.float64)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    to_origin = to_origin.astype(np.float64)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3).astype(np.float64)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=debug_dir or "debug",
        debug=int(debug),
        glctx=glctx,
    )

    camera = cam_main["cam"]
    if frames is None:
        frames = camera.get_frames()
        if frames is None:
            raise RuntimeError("Failed to retrieve frames from camera")

    color_bgr = np.asarray(frames["color"])
    depth_raw = np.asarray(frames["depth"], dtype=np.float32)
    depth_scale = float(getattr(camera, "depth_scale", 0.001))
    depth_m = depth_raw * depth_scale

    mask = _generate_mask(color_bgr, target, mask_mode, qwen_model_path, mask_kwargs)
    mask_binary = (mask > 0).astype(np.uint8)

    cam_k = np.asarray(cam_main["cam_k"], dtype=np.float64)

    pose = estimator.register(
        K=cam_k,
        rgb=color_bgr,
        depth=depth_m,
        ob_mask=mask_binary,
        iteration=iteration,
    )
    center_pose = pose @ np.linalg.inv(to_origin)

    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    vis = draw_posed_3d_box(cam_k, img=color_rgb, ob_in_cam=center_pose, bbox=bbox)
    vis = draw_xyz_axis(
        color_rgb,
        ob_in_cam=center_pose,
        scale=0.1,
        K=cam_k,
        thickness=3,
        transparency=0,
        is_input_rgb=True,
    )
    visualization_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    if debug:
        debug_path = debug_dir or os.path.join("debug", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(debug_path, exist_ok=True)
        cv2.imwrite(os.path.join(debug_path, "color.png"), color_bgr)
        cv2.imwrite(os.path.join(debug_path, "mask.png"), mask)
        cv2.imwrite(os.path.join(debug_path, "pose_vis.png"), visualization_bgr)

    logger.info("FoundationPose detection completed for target '%s'", target)
    return DetectionResult(
        target=target,
        center_pose=center_pose,
        raw_pose=pose,
        to_origin=to_origin,
        bbox=bbox,
        mask=mask,
        color_bgr=color_bgr,
        depth_m=depth_m,
        visualization_bgr=visualization_bgr,
    )


def plan_grasp_pose(
    target: str,
    center_pose: np.ndarray,
    cam_main: Dict[str, Any],
    robot_main: Dict[str, Any],
    *,
    grasp_library_path: Optional[str] = None,
) -> GraspPlan:
    """Generate a grasp plan from the detected object pose."""

    grasp_library = _load_grasp_library(grasp_library_path)
    if target not in grasp_library:
        raise KeyError(f"Target '{target}' not found in grasp library")

    grasp_params = grasp_library[target]
    z_xoy_angle = float(grasp_params.get("z_xoy_angle", 0.0))
    vertical_euler = grasp_params.get("vertical_euler", [-180.0, 0.0, -90.0])
    grasp_tilt_angle = float(grasp_params.get("grasp_tilt_angle", 0.0))
    angle_threshold = float(grasp_params.get("angle_threshold", 5.0))
    T_safe_distance = float(grasp_params.get("T_safe_distance", 0.0))
    z_safe_distance = float(grasp_params.get("z_safe_distance", 0.0))
    gripper_close_pos = int(grasp_params.get("gripper_close_pos", 80))

    T_ee_cam = cam_main.get("T_ee_cam") or cam_main.get("Rt")
    if T_ee_cam is None:
        raise KeyError("Camera calibration (T_ee_cam) is required in cam_main")

    T_tcp_ee_offset = float(robot_main.get("tcp2ee", 0.0))
    robot = robot_main["robot"]

    T_cam_object = SE3(center_pose, check=False)
    pose_now = robot.get_pose()
    x_e, y_e, z_e, rx_e, ry_e, rz_e = pose_now

    T_base_ee = SE3.Rt(
        SO3.RPY([rx_e, ry_e, rz_e], unit="deg", order="zyx"),
        np.array([x_e, y_e, z_e]) / 1000.0,
        check=False,
    )

    T_base_cam = T_base_ee * T_ee_cam
    T_base_obj = T_base_cam * T_cam_object

    T_base_obj_array = np.array(T_base_obj, dtype=float)
    current_rotation_matrix = T_base_obj_array[:3, :3]
    current_z_axis = current_rotation_matrix[:, 2]
    target_z_axis = np.array([0.0, 0.0, 1.0])
    z_angle_error = np.degrees(
        np.arccos(np.clip(np.dot(current_z_axis, target_z_axis), -1.0, 1.0))
    )

    if z_angle_error > angle_threshold:
        rotation_axis = np.cross(current_z_axis, target_z_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        if rotation_axis_norm >= 1e-6:
            rotation_axis /= rotation_axis_norm
            rotation_angle = np.arccos(
                np.clip(np.dot(current_z_axis, target_z_axis), -1.0, 1.0)
            )
            K = np.array(
                [
                    [0, -rotation_axis[2], rotation_axis[1]],
                    [rotation_axis[2], 0, -rotation_axis[0]],
                    [-rotation_axis[1], rotation_axis[0], 0],
                ]
            )
            R_z_align = (
                np.eye(3)
                + np.sin(rotation_angle) * K
                + (1 - np.cos(rotation_angle)) * (K @ K)
            )
            current_rotation_matrix = R_z_align @ current_rotation_matrix

    rotation_matrix_after_z = current_rotation_matrix
    current_x_axis = rotation_matrix_after_z[:, 0]
    x_projected = np.array([current_x_axis[0], current_x_axis[1], 0.0])
    x_norm = np.linalg.norm(x_projected)
    if x_norm > 1e-6:
        x_projected /= x_norm
        x_angle = np.arctan2(x_projected[1], x_projected[0])
        R_z_align_xy = np.array(
            [
                [np.cos(-x_angle), -np.sin(-x_angle), 0],
                [np.sin(-x_angle), np.cos(-x_angle), 0],
                [0, 0, 1],
            ]
        )
        rotation_matrix_after_z = R_z_align_xy @ rotation_matrix_after_z

    theta = np.radians(z_xoy_angle)
    R_z = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    current_rotation_matrix = R_z @ rotation_matrix_after_z

    T_base_obj_final = np.eye(4)
    T_base_obj_final[:3, :3] = current_rotation_matrix
    T_base_obj_final[:3, 3] = T_base_obj_array[:3, 3]
    T_base_obj_final = SE3(T_base_obj_final, check=False)

    tilted_euler = [
        float(vertical_euler[0]) + grasp_tilt_angle,
        float(vertical_euler[1]),
        float(vertical_euler[2]),
    ]
    R_target_xyz = R.from_euler("xyz", tilted_euler, degrees=True)
    T_object_grasp_ideal = SE3.Rt(
        SO3(R_target_xyz.as_matrix()),
        [0, 0, 0],
        check=False,
    )

    T_base_grasp_ideal = T_base_obj_final * T_object_grasp_ideal
    T_tcp_ee = SE3(0, 0, T_tcp_ee_offset)
    T_safe = SE3(0, 0, T_safe_distance)
    T_base_ee_ideal = T_base_grasp_ideal * T_tcp_ee * T_safe

    pos_mm = T_base_ee_ideal.t * 1000.0
    rx, ry, rz = T_base_ee_ideal.rpy(unit="deg", order="zyx")
    rz = normalize_angle(rz)

    pre_grasp_pose = (
        float(pos_mm[0]),
        float(pos_mm[1]),
        float(pos_mm[2] + z_safe_distance),
        float(rx),
        float(ry),
        float(rz),
    )
    grasp_pose = (
        float(pos_mm[0]),
        float(pos_mm[1]),
        float(pos_mm[2]),
        float(rx),
        float(ry),
        float(rz),
    )

    return GraspPlan(
        target=target,
        pre_grasp_pose=pre_grasp_pose,
        grasp_pose=grasp_pose,
        z_safe_distance=z_safe_distance,
        gripper_close_pos=gripper_close_pos,
        T_base_ee_ideal=T_base_ee_ideal,
        parameters=grasp_params,
    )


def choose_grasp_pose(*args, **kwargs):
    """Backward-compatible wrapper returning tuple poses."""
    plan = plan_grasp_pose(*args, **kwargs)
    return list(plan.pre_grasp_pose), list(plan.grasp_pose)


def execute_grasp_plan(
    robot_main: Dict[str, Any],
    gripper: Optional[Any],
    grasp_plan: GraspPlan,
    *,
    approach_speed: float = 15.0,
    approach_acc: float = 15.0,
    grasp_speed: float = 8.0,
    grasp_acc: float = 8.0,
    retreat_speed: Optional[float] = None,
    retreat_acc: Optional[float] = None,
    gripper_open_position: Optional[int] = None,
    gripper_close_force: float = 80.0,
    gripper_close_speed: float = 30.0,
    hold_duration: float = 0.5,
    retreat_offset: Optional[float] = None,
    logger: logging.Logger = default_logger,
) -> bool:
    """Execute a grasp plan with the robot and gripper."""

    robot = robot_main["robot"]
    retreat_speed = retreat_speed or approach_speed
    retreat_acc = retreat_acc or approach_acc
    retreat_offset = retreat_offset if retreat_offset is not None else grasp_plan.z_safe_distance

    try:
        if gripper and gripper_open_position is not None:
            gripper.control(position=int(gripper_open_position), force=30, speed=40)
            time.sleep(0.2)

        robot.move_to_pose(*grasp_plan.pre_grasp_pose, speed=approach_speed, acceleration=approach_acc)
        robot.move_to_pose(*grasp_plan.grasp_pose, speed=grasp_speed, acceleration=grasp_acc)

        if gripper:
            gripper.control(
                position=int(grasp_plan.gripper_close_pos),
                force=int(np.clip(gripper_close_force, 0, 100)),
                speed=int(np.clip(gripper_close_speed, 0, 100)),
            )
            time.sleep(max(hold_duration, 0.1))

        retreat_pose = list(grasp_plan.grasp_pose)
        retreat_pose[2] += retreat_offset
        robot.move_to_pose(*retreat_pose, speed=retreat_speed, acceleration=retreat_acc)
        return True

    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("Failed to execute grasp plan: %s", exc)
        return False


def detect_glassbar_direction(
    angle_camera,
    *,
    save_dir: Optional[str] = None,
    max_attempts: int = 100,
    sleep_interval: float = 0.1,
    logger: logging.Logger = default_logger,
) -> DirectionDetectionResult:
    """Detect the orientation of the glass bar using the angle camera."""

    attempts = 0
    last_angles: Iterable[float] = []
    avg_angle = 0.0
    start_time = time.time()

    while attempts < max_attempts:
        attempts += 1
        raw_image = angle_camera.get_current_frame()
        if raw_image is None:
            logger.debug("Angle camera frame unavailable, retrying (%d/%d)", attempts, max_attempts)
            time.sleep(sleep_interval)
            continue

        logger.debug("Detecting glass bar orientation (attempt %d)", attempts)
        last_angles, avg_angle = detect_dent_orientation(raw_image, save_dir=save_dir)
        if last_angles:
            logger.info(
                "检测到物体朝向角度: %s, 平均: %.2f°, 绝对值: %.2f°",
                last_angles,
                avg_angle,
                abs(avg_angle),
            )
            break

        logger.info("当前图像未检测到明显方向特征，继续等待…")
        time.sleep(sleep_interval)

    return DirectionDetectionResult(
        angles=last_angles,
        avg_angle=avg_angle,
        attempts=attempts,
        timestamp=start_time,
    )


def adjust_to_vertical_and_lift(
    robot,
    avg_angle: float,
    grasp_tilt_angle: float,
    *,
    x_adjustment: float = 115.0,
    z_adjustment: float = 180.0,
    adjust_speed: float = 12.0,
    wait_time: Optional[float] = 3.0,
    logger: logging.Logger = default_logger,
) -> Dict[str, Any]:
    """Adjust the glass bar orientation to vertical and lift to a safe height."""

    logger.info("开始调整玻璃棒姿态至垂直桌面向下")
    pose_now = robot.get_pose()
    logger.debug(
        "当前位姿: x=%.2f, y=%.2f, z=%.2f, rx=%.2f, ry=%.2f, rz=%.2f",
        *pose_now,
    )

    delta_ee = avg_angle - grasp_tilt_angle
    pose_target = [
        pose_now[0] + x_adjustment,
        pose_now[1],
        pose_now[2] + z_adjustment,
        pose_now[3] + delta_ee,
        pose_now[4],
        pose_now[5],
    ]

    logger.info(
        "目标位姿: x=%.2f, y=%.2f, z=%.2f, rx=%.2f, ry=%.2f, rz=%.2f",
        *pose_target,
    )

    robot.move_to_pose(
        pose_target[0],
        pose_target[1],
        pose_target[2],
        pose_target[3],
        pose_target[4],
        pose_target[5],
        speed=int(round(adjust_speed)),
        acceleration=1,
    )

    if wait_time is None and adjust_speed > 0:
        wait_time = 1.0 / adjust_speed
    if wait_time:
        rospy.Rate(1.0 / wait_time).sleep()

    pose_after_adjust = robot.get_pose()
    angle_error = abs(pose_after_adjust[3] - pose_target[3])

    logger.info(
        "调整后姿态: rx=%.2f° (目标 %.2f°), 角度误差 %.2f°",
        pose_after_adjust[3],
        pose_target[3],
        angle_error,
    )

    success = angle_error < 5.0
    return {
        "success": success,
        "initial_pose": pose_now,
        "target_pose": pose_target,
        "final_pose": pose_after_adjust,
        "delta_angle": delta_ee,
        "x_adjustment": x_adjustment,
        "z_adjustment": z_adjustment,
        "angle_error": angle_error,
    }


def descend_with_force_feedback(
    robot,
    *,
    move_step: float = 1.0,
    max_steps: int = 700,
    force_threshold: float = 1.0,
    sample_interval: float = 0.03,
    max_force_samples: int = 30,
    consecutive_hits_required: int = 2,
    speed: float = 5.0,
    logger: logging.Logger = default_logger,
) -> ContactDetectionResult:
    """Descend along Z until contact is detected via force sensor."""

    logger.info("开始垂直下降并监测力反馈…")
    pose_initial = robot.get_pose()
    pose_current = pose_initial.copy()

    contact_detected = False
    contact_force = 0.0
    steps_taken = 0

    for step in range(max_steps):
        rospy.Rate(33).sleep()
        pose_current[2] -= move_step
        robot.move_to_pose(
            pose_current[0],
            pose_current[1],
            pose_current[2],
            pose_current[3],
            pose_current[4],
            pose_current[5],
            speed=int(speed),
            acceleration=1,
        )

        steps_taken = step + 1
        consecutive_hits = 0

        for _ in range(max_force_samples):
            rospy.Rate(1 / sample_interval).sleep()
            force_values = robot.get_force()
            if not force_values:
                continue

            max_force_component = max(abs(value) for value in force_values)
            if max_force_component >= force_threshold:
                consecutive_hits += 1
                contact_force = max_force_component
                if consecutive_hits >= consecutive_hits_required:
                    contact_detected = True
                    logger.info(
                        "检测到显著力反馈，步数 %d, 力 %.2fN",
                        steps_taken,
                        contact_force,
                    )
                    break
            else:
                consecutive_hits = 0

        if contact_detected:
            break

    pose_final = robot.get_pose()
    descent_distance = steps_taken * move_step
    if not contact_detected:
        logger.warning("达到最大步数，未检测到明显接触")

    return ContactDetectionResult(
        contact_detected=contact_detected,
        steps_taken=steps_taken,
        descent_distance=descent_distance,
        final_pose=pose_final,
    )


def force_guided_spiral_insertion(
    robot,
    *,
    descent_step: float = 1.0,
    retract_distance: float = 4.0,
    max_descent: float = 37.0,
    force_threshold: float = 2.0,
    samples_per_check: int = 6,
    sample_interval: float = 0.03,
    spiral_step: float = 1.1,
    spiral_angle_increment_deg: float = 20.0,
    max_spiral_radius: float = 60.0,
    planar_speed: float = 1.0,
    descent_speed: float = 1.0,
    verbose: bool = False,
    logger: logging.Logger = default_logger,
) -> ForceInsertionResult:
    """Perform force-guided insertion with XY spiral micro-adjustments."""

    if descent_step <= 0 or max_descent <= 0:
        raise ValueError("descent_step 和 max_descent 必须为正数")

    initial_pose = robot.get_pose()
    current_pose = initial_pose.copy()
    initial_z = current_pose[2]

    planar_attempts = 0
    planar_successes = 0
    contact_events = []
    cumulative_spiral_points = 0

    if verbose:
        logger.info(
            "开始力控插入: 初始Z %.2f, 每步下降 %.2f, 最大下降 %.2f, 力阈值 %.2f",
            initial_z,
            descent_step,
            max_descent,
            force_threshold,
        )

    while True:
        current_pose = robot.get_pose()
        current_descent = initial_z - current_pose[2]
        if current_descent >= max_descent - 1e-2:
            if verbose:
                logger.info("达到目标深度 %.2f mm", max_descent)
            break

        target_pose = current_pose.copy()
        target_pose[2] -= descent_step
        robot.move_to_pose(
            target_pose[0],
            target_pose[1],
            target_pose[2],
            target_pose[3],
            target_pose[4],
            target_pose[5],
            speed=int(descent_speed),
            acceleration=1,
        )

        measured_force = measure_max_force(robot, samples=samples_per_check, interval=sample_interval)
        new_pose = robot.get_pose()
        new_descent = initial_z - new_pose[2]

        if verbose:
            logger.info("[下降] 深度 %.2f/%.2f mm -> 力 %.2f N", new_descent, max_descent, measured_force)

        if measured_force < force_threshold:
            continue

        contact_events.append({"depth": new_descent, "force": measured_force})
        if verbose:
            logger.info("检测到力反馈，开始XY螺旋微调")

        safe_pose = robot.get_pose()
        safe_pose[2] += retract_distance
        robot.move_to_pose(
            safe_pose[0],
            safe_pose[1],
            safe_pose[2],
            safe_pose[3],
            safe_pose[4],
            safe_pose[5],
            speed=int(descent_speed),
            acceleration=1,
        )
        rospy.Rate(1 / 1.5).sleep()
        safe_pose = robot.get_pose()

        spiral_center = initial_pose.copy()
        spiral_center[2] = safe_pose[2]
        spiral_center[3:] = safe_pose[3:]

        success = False
        attempts = 0
        for dx, dy in generate_spiral_offsets(
            step=spiral_step,
            max_radius=max_spiral_radius,
            angle_increment_deg=spiral_angle_increment_deg,
            skip_count=cumulative_spiral_points,
        ):
            attempts += 1
            planar_attempts += 1
            cumulative_spiral_points += 1

            current_pose = robot.get_pose()
            target_pose = current_pose.copy()
            target_pose[0] += dx
            target_pose[1] += dy
            robot.move_to_pose(
                target_pose[0],
                target_pose[1],
                target_pose[2],
                target_pose[3],
                target_pose[4],
                target_pose[5],
                speed=int(planar_speed),
                acceleration=1,
            )
            rospy.Rate(1 / 0.4).sleep()

            probe_pose = robot.get_pose()
            probe_pose[2] -= descent_step
            robot.move_to_pose(
                probe_pose[0],
                probe_pose[1],
                probe_pose[2],
                probe_pose[3],
                probe_pose[4],
                probe_pose[5],
                speed=int(descent_speed),
                acceleration=1,
            )
            rospy.Rate(1 / 0.4).sleep()

            measured_force = measure_max_force(robot, samples=samples_per_check, interval=sample_interval)
            if verbose:
                logger.info("    [微调] 偏移(%.2f, %.2f) -> 力 %.2fN", dx, dy, measured_force)

            if measured_force < force_threshold:
                success = True
                planar_successes += 1
                break

            recovery_pose = robot.get_pose()
            recovery_pose[2] = spiral_center[2]
            robot.move_to_pose(
                recovery_pose[0],
                recovery_pose[1],
                recovery_pose[2],
                recovery_pose[3],
                recovery_pose[4],
                recovery_pose[5],
                speed=int(descent_speed),
                acceleration=1,
            )
            rospy.Rate(1 / 0.8).sleep()

        if not success:
            final_pose = robot.get_pose()
            logger.warning("螺旋微调失败，提前结束插入流程")
            return ForceInsertionResult(
                success=False,
                total_descent=initial_z - final_pose[2],
                planar_attempts=planar_attempts,
                planar_successes=planar_successes,
                contact_events=contact_events,
                cumulative_spiral_points=cumulative_spiral_points,
                final_pose=final_pose,
            )

    final_pose = robot.get_pose()
    total_descent = initial_z - final_pose[2]
    success = total_descent >= max_descent - 1e-3

    return ForceInsertionResult(
        success=success,
        total_descent=total_descent,
        planar_attempts=planar_attempts,
        planar_successes=planar_successes,
        contact_events=contact_events,
        cumulative_spiral_points=cumulative_spiral_points,
        final_pose=final_pose,
    )


def monitor_contact_with_camera(
    contact_camera,
    robot,
    *,
    sample_interval: float = 0.1,
    move_step: float = 3.0,
    max_steps: int = 700,
    change_threshold: float = 3.0,
    pixel_threshold: int = 2,
    min_area: int = 2,
    debug_dir: Optional[str] = None,
    wait_rate_hz: float = 33.0,
    on_debug: Optional[Callable[[np.ndarray, np.ndarray, int], None]] = None,
    logger: logging.Logger = default_logger,
) -> ContactDetectionResult:
    """Monitor contact using a vision-based camera difference method."""

    rate_wait = rospy.Rate(wait_rate_hz)
    initial_frame = None
    while initial_frame is None:
        initial_frame = contact_camera.get_current_frame()
        if initial_frame is None:
            logger.info("等待接触相机初始图像…")
            rospy.sleep(sample_interval)

    logger.info("已获取初始图像，开始逐步下降检测接触")
    pose_current = robot.get_pose()
    steps_taken = 0

    for step in range(max_steps):
        rate_wait.sleep()
        frame_before = contact_camera.get_current_frame()
        if frame_before is None:
            logger.debug("步骤 %d 缺少动作前图像，继续等待", step + 1)
            rospy.sleep(sample_interval)
            continue

        pose_current[2] -= move_step
        robot.move_to_pose(
            pose_current[0],
            pose_current[1],
            pose_current[2],
            pose_current[3],
            pose_current[4],
            pose_current[5],
            speed=5,
            acceleration=1,
        )

        frame_after = None
        has_change = False
        for _ in range(int(max(1, sample_interval * 20))):
            rospy.sleep(sample_interval)
            candidate = contact_camera.get_current_frame()
            if candidate is None:
                continue
            frame_after = candidate
            has_change = contact_camera.has_significant_change(
                frame_before,
                frame_after,
                change_threshold=change_threshold,
                pixel_threshold=pixel_threshold,
                min_area=min_area,
                save_dir=debug_dir,
                step_num=step,
            )
            if has_change:
                break

        steps_taken = step + 1
        if frame_after is not None and on_debug:
            on_debug(frame_before, frame_after, step)

        if frame_after is None:
            logger.debug("步骤 %d 未取得动作后图像", step + 1)
            continue

        if has_change:
            logger.info(
                "检测到显著变化，可能已接触桌面 (步数 %d, 下降 %.2fmm)",
                step + 1,
                (step + 1) * move_step,
            )
            break

    else:
        logger.warning("达到最大移动距离，未检测到明显变化")

    pose_final = robot.get_pose()
    descent_distance = steps_taken * move_step
    return ContactDetectionResult(
        contact_detected=steps_taken < max_steps,
        steps_taken=steps_taken,
        descent_distance=descent_distance,
        final_pose=pose_final,
    )

