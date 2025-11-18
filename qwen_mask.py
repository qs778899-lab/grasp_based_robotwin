import os
import json
import re
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from ultralytics import SAM


_global_qwen_model: Optional[Qwen3VLForConditionalGeneration] = None
_global_qwen_processor: Optional[AutoProcessor] = None
_global_qwen_device: Optional[torch.device] = None
_global_qwen_model_id: Optional[str] = None
_global_sam_predictor: Optional[SAM] = None


def _resolve_model_path(model_path: Optional[str]) -> Tuple[str, bool]:
    if model_path is None:
        model_path = os.environ.get(
            "QWEN_VL_MODEL_PATH",
            "/home/erlin/work/labgrasp/Qwen3-VL/Qwen3-VL-4B-Instruct",
        )
    local_files_only = os.path.isdir(model_path)
    if local_files_only:
        model_path = os.path.abspath(model_path)
    return model_path, local_files_only


def _ensure_qwen3_model(model_path: Optional[str] = None):
    global _global_qwen_model, _global_qwen_processor, _global_qwen_device, _global_qwen_model_id

    resolved_path, local_files_only = _resolve_model_path(model_path)
    if (
        _global_qwen_model is not None
        and _global_qwen_model_id == resolved_path
    ):
        return

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    _global_qwen_device = torch.device(device_str)
    _global_qwen_model_id = resolved_path

    load_kwargs: Dict[str, Any] = {
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }
    try:
        _global_qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            resolved_path,
            torch_dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            **load_kwargs,
        )
    except TypeError:
        _global_qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
            resolved_path,
            dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
            **load_kwargs,
        )
    _global_qwen_model.eval()

    _global_qwen_processor = AutoProcessor.from_pretrained(
        resolved_path,
        **load_kwargs,
    )
    print(f"Qwen3-VL模型已加载: {resolved_path}")


def _ensure_sam_predictor():
    global _global_sam_predictor
    if _global_sam_predictor is None:
        _global_sam_predictor = SAM(
            model="/home/erlin/TCP-IP-Python-V4/sam2.1_b.pt"
        )
        print("SAM模型已加载")


def _build_messages(image: Image.Image, prompt: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": (
                        "You are provided an image. "
                        f"Please locate the object described as \"{prompt}\". "
                        "Return the result strictly as JSON with a list of bounding boxes. "
                        "Example: "
                        "[{\"bbox_2d\": [x1, y1, x2, y2]}]. "
                        "Coordinates must be in pixel units within the input image size."
                    ),
                },
            ],
        }
    ]


def _extract_json_payload(output_text: str) -> Optional[str]:
    if not output_text:
        return None

    matches = re.findall(r"```json\s*(.*?)\s*```", output_text, re.DOTALL)
    if matches:
        return matches[0]

    bracket_match = re.search(r"\[\s*{.*}\s*\]", output_text, re.DOTALL)
    if bracket_match:
        return bracket_match.group(0)

    inline_match = re.search(r"\{[^{}]*bbox[^{}]*\}", output_text, re.DOTALL)
    if inline_match:
        return f"[{inline_match.group(0)}]"

    return None


def _scale_bbox(bbox: List[float], image_size: Tuple[int, int]) -> Optional[List[int]]:
    h, w = image_size
    x1, y1, x2, y2 = bbox
    values = np.array([x1, y1, x2, y2], dtype=float)

    if np.max(values) <= 1.5:
        values[0::2] *= w
        values[1::2] *= h
    elif np.max(values) > max(w, h):
        values[0::2] *= w / 1000.0
        values[1::2] *= h / 1000.0

    x1, y1, x2, y2 = values
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    x1 = int(np.clip(x1, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y2 = int(np.clip(y2, 0, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _parse_bounding_box(
    output_text: str, image_size: Tuple[int, int]
) -> Optional[List[int]]:
    json_payload = _extract_json_payload(output_text)
    if not json_payload:
        print("Qwen输出中未找到JSON格式的边界框。")
        return None

    try:
        data = json.loads(json_payload)
    except json.JSONDecodeError as exc:
        print(f"解析Qwen输出失败: {exc}")
        return None

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return None

    for item in data:
        if not isinstance(item, dict):
            continue
        bbox = item.get("bbox_2d") or item.get("bbox") or item.get("box")
        if not bbox or len(bbox) < 4:
            continue
        scaled = _scale_bbox(bbox, image_size)
        if scaled:
            return scaled

    print("未能从Qwen输出中提取有效的边界框。")
    return None


def _predict_bbox_with_qwen(
    image: np.ndarray,
    prompt: str,
    model_path: Optional[str] = None,
) -> Optional[List[int]]:
    _ensure_qwen3_model(model_path=model_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    messages = _build_messages(pil_image, prompt)
    chat_template = _global_qwen_processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _global_qwen_processor(
        text=[chat_template],
        images=[pil_image],
        return_tensors="pt",
    )
    inputs = {k: v.to(_global_qwen_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = _global_qwen_model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.01,
        )

    generated_ids_trimmed = [
        out[len(inp) :] for inp, out in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = _global_qwen_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    print(f"Qwen输出: {output_text}")

    h, w = image.shape[:2]
    return _parse_bounding_box(output_text, (h, w))


def _segment_with_sam(image: np.ndarray, bbox: List[int]) -> np.ndarray:
    _ensure_sam_predictor()
    results = _global_sam_predictor(image, bboxes=[bbox])
    mask = results[0].masks.data[0].cpu().numpy()
    mask = (mask > 0).astype(np.uint8) * 255
    return mask


def get_mask_from_qwen(
    rgb_image: np.ndarray,
    prompt: str,
    model_path: Optional[str] = None,
    bbox_vis_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    使用本地Qwen3-VL模型检测指定目标的包围框，并通过SAM生成掩码。
    Args:
        rgb_image: BGR格式图像 (H, W, 3)
        prompt: 目标描述文本
        model_path: 模型路径（可选，默认读取环境变量或固定路径）
    Returns:
        掩码图 (H, W)，uint8格式，或在检测失败时返回None
    """
    if rgb_image is None or rgb_image.size == 0:
        print("输入图像为空，无法生成掩码。")
        return None

    bbox = _predict_bbox_with_qwen(
        rgb_image,
        prompt,
        model_path=model_path,
    )
    if bbox is None:
        return None

    debug_image = rgb_image.copy()
    cv2.rectangle(debug_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    cv2.imshow("qwen_bbox", debug_image)
    cv2.waitKey(1)
    if bbox_vis_path:
        cv2.imwrite(bbox_vis_path, debug_image)

    return _segment_with_sam(rgb_image, bbox)

