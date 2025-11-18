"""
Author: zhangcongshe

Date: 2025/11/1

Version: 1.0
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import nvdiffrast.torch as dr
import trimesh
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3, SO3

from FoundationPose.estimater import FoundationPose
from FoundationPose.learning.training.predict_score import ScorePredictor
from FoundationPose.learning.training.predict_pose_refine import PoseRefinePredictor

from Utils import (
    draw_posed_3d_box,
    draw_xyz_axis,
    set_logging_format,
    set_seed,
)

from dino_mask import get_mask_from_GD
from qwen_mask import get_mask_from_qwen

logger = logging.getLogger(__name__)

GRASP_LIBRARY_PATH = os.path.join(os.path.dirname(__file__), "GraspLibrary.json")
_GRASP_LIBRARY_CACHE: Optional[Tuple[str, Dict[str, Any]]] = None


@dataclass
class DetectionResult:
    center_pose: np.ndarray
    raw_pose: np.ndarray
    to_origin: np.ndarray
    bbox: np.ndarray
    mask: np.ndarray
    color_bgr: np.ndarray
    depth_m: np.ndarray
    debug_visual_bgr: Optional[np.ndarray] = None


@dataclass
class GraspPlan:
    target: str
    pre_grasp_pose: Tuple[float, float, float, float, float, float]
    grasp_pose: Tuple[float, float, float, float, float, float]
    z_safe_distance: float
    gripper_close_pos: int
    T_base_ee_ideal: SE3
    parameters: Dict[str, Any]


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
    if mask_mode.lower() in {"qwen", "qwen3", "vl"}:
        mask = get_mask_from_qwen(color_bgr, target, model_path=qwen_model_path, **mask_kwargs)
    elif mask_mode.lower() in {"groundingdino", "dino", "gd"}:
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
) -> DetectionResult:
    """Detect the object pose using FoundationPose and segmentation masks."""

    mask_kwargs = mask_kwargs or {}
    set_logging_format()
    set_seed(0)

    mesh = trimesh.load(mesh_path, force="mesh")
    mesh = mesh.copy()
    mesh.vertices /= 1000.0
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

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

    cam_k = np.asarray(cam_main["cam_k"], dtype=np.float32)

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
    debug_visual_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    if debug:
        debug_path = debug_dir or os.path.join("debug", time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(debug_path, exist_ok=True)
        cv2.imwrite(os.path.join(debug_path, "color.png"), color_bgr)
        cv2.imwrite(os.path.join(debug_path, "mask.png"), mask)
        cv2.imwrite(os.path.join(debug_path, "pose_vis.png"), debug_visual_bgr)

    logger.info("FoundationPose detection completed for target '%s'", target)
    return DetectionResult(
        center_pose=center_pose,
        raw_pose=pose,
        to_origin=to_origin,
        bbox=bbox,
        mask=mask,
        color_bgr=color_bgr,
        depth_m=depth_m,
        debug_visual_bgr=debug_visual_bgr,
    )


def choose_grasp_pose(
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

    x_axis = current_rotation_matrix[:, 0]
    x_projected = np.array([x_axis[0], x_axis[1], 0.0])
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
        current_rotation_matrix = R_z_align_xy @ current_rotation_matrix

    theta = np.radians(z_xoy_angle)
    R_z = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    current_rotation_matrix = R_z @ current_rotation_matrix

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

    def _normalize_angle(angle: float) -> float:
        angle = angle % 360.0
        if angle > 180.0:
            return angle - 360.0
        return angle

    rz = _normalize_angle(rz)

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

