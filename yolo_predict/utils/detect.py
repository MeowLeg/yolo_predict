import os
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from cv2.typing import MatLike
from ultralytics.engine.results import Results
from ultralytics.models.yolo.model import YOLO

# ---------------数据结构-----------------


@dataclass
class Prediction:
    tag: str
    title: str
    content: str
    model: str
    pipe: str  # 没用，不再用redis
    imgsz: int
    labels: list[str]
    conf: float
    iou: float
    split: int
    yolo: YOLO
    pre_conditions: List[str]


@dataclass
class Image:
    id: int
    path: str
    uuid: str
    code: str
    create_date: str


@dataclass
class Alert:
    uuid: str
    title: str
    content: str
    datetime: str
    alert_type: list[str]
    mediaUrl: str
    videoUrl: list[str]
    otherUrl: list[str]


# ---------------方法定义-----------------


def __split_image(
    image: MatLike, n: int = 2
) -> tuple[list[MatLike], list[tuple[int, int]]]:
    # 分割图片
    shape: tuple[int, int, int] = image.shape  # pyright: ignore[reportAssignmentType]
    h, w = shape[:2]
    sub_h, sub_w = h // n, w // n
    sub_images: list[MatLike] = []
    positions: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            y1 = i * sub_h
            y2 = (i + 1) * sub_h if i < n - 1 else h
            x1 = j * sub_w
            x2 = (j + 1) * sub_w if j < n - 1 else w
            sub_img = image[y1:y2, x1:x2]
            sub_images.append(sub_img)
            positions.append((x1, y1))
    return sub_images, positions


def __detect_split_image(
    model: YOLO, image: MatLike, conf: float = 0.5, iou: float = 0.45
):
    # 对分割的图片进行检测
    sub_images, positions = __split_image(image, n=2)
    all_boxes: list[list[int | float]] = []

    for sub_img, (x_offset, y_offset) in zip(sub_images, positions):
        results: list[Results] = model(sub_img, conf=conf, iou=iou)  # pyright: ignore[reportUnknownVariableType]
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            # xyxy: tensor([[730.1935, 276.2485, 804.1757, 315.5241]])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # pyright: ignore[reportAny, reportUnknownMemberType]
            x1 += x_offset  # pyright: ignore[reportAny]
            y1 += y_offset  # pyright: ignore[reportAny]
            x2 += x_offset  # pyright: ignore[reportAny]
            y2 += y_offset  # pyright: ignore[reportAny]

            cls: int = int(box.cls[0])  # pyright: ignore[reportUnknownMemberType]
            box_conf: float = float(box.conf[0])  # pyright: ignore[reportUnknownMemberType]
            all_boxes.append([x1, y1, x2, y2, box_conf, cls])

    if not all_boxes:
        return np.array([])
    n_all_boxes = np.array(all_boxes)
    boxes = n_all_boxes[:, :4]
    scores = n_all_boxes[:, 4]
    nms_indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf, iou)  # pyright: ignore[reportAny]
    if len(nms_indices) == 0:
        return np.array([])
    indices = (  # pyright: ignore[reportUnknownVariableType]
        nms_indices.flatten() if isinstance(nms_indices, np.ndarray) else nms_indices[0]
    )
    return n_all_boxes[indices]  # pyright: ignore[reportAny]


def __nms_next(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    # 返回所有n2中与n1有交集的元素
    if not n1.size or not n2.size:
        return np.array([])
    n1_expanded = n1[np.newaxis, :, :]
    n2_expanded = n1[:, np.newaxis, :]
    intersects = (
        (n1_expanded[..., 0] < n2_expanded[..., 2])  # A.x1 < B.x2
        & (n1_expanded[..., 2] > n2_expanded[..., 0])  # A.x2 > B.x1
        & (n1_expanded[..., 1] < n2_expanded[..., 3])  # A.y1 < B.y2
        & (n1_expanded[..., 3] > n2_expanded[..., 1])  # A.y2 > B.y1
    )
    has_intersection_indices = intersects.any(axis=1)
    return n2[has_intersection_indices]


# todo
# prediction应该是一个数组，互相之间是与操作
# 也就是说第一个预测失败的话就直接可以停止了
# 如果两个预测都是有数据，则两者的框做nms，这里的iou可以非常小，只要有交接即可
# 最后一个预测的label作为整个组的标签
def predict(
    predictions: List[Prediction],
    out_path: str,
    im_path: str,
) -> bool:
    image: MatLike | None = cv2.imread(im_path)
    if image is None:
        return False

    merged_boxes = np.array([])
    for idx, p in enumerate(predictions):
        _boxes = __detect_split_image(p.yolo, image, p.conf, p.iou)
        if _boxes.shape[0] == 0:
            return False
        elif idx > 0:
            merged_boxes = __nms_next(merged_boxes, _boxes)
            if merged_boxes.shape[0] == 0:
                return False
        else:
            merged_boxes = _boxes

    pdct = predictions[-1]  # 最后一个预测对象为任务对象，前面的都是前置条件
    detected = False
    for box in merged_boxes:  # pyright: ignore[reportAny]
        # float, float, float, float, float, int
        x1, y1, x2, y2, conf, cls = box  # pyright: ignore[reportAny]
        _ = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # pyright: ignore[reportAny]

        cate_label = pdct.yolo.names[int(cls)]
        if cate_label == pdct.tag and not detected:
            # 仅当标签与预测标签相同时才添加到结果列表中
            detected = True

        # 绘框
        label = f"{pdct.yolo.names[int(cls)]} {conf: .2f}"  # pyright: ignore[reportAny]
        y1_text = int(y1) - 10  # pyright: ignore[reportAny]
        if y1_text <= 0:
            y1_text = 0
        _ = cv2.putText(
            image,
            label,
            (int(x1), y1_text),  # pyright: ignore[reportAny]
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        _ = cv2.imwrite(os.path.join(out_path, os.path.basename(im_path)), image)

    return detected
