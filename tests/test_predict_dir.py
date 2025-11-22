import os
from typing import Any

import tomli
from ultralytics.models.yolo.model import YOLO

from yolo_predict.utils.detect import Prediction, predict


def __filter_model(code: str, cfg_predicts: list[Prediction]) -> Prediction | None:
    # 列出实现的模型
    for p in cfg_predicts:
        if p.tag == code:
            return p


def predict_dir(code: str) -> None:
    # 用于测试
    with open("./config.toml", "rb") as f:
        cfg: dict[str, Any] = tomli.load(f)  #  pyright: ignore[reportAny]
        dataclass_prediction_list = []
        for p in cfg["predict"]:
            p["yolo"] = YOLO(p["model"])
            dataclass_prediction_list.append(Prediction(**p))
        cfg["predict"] = dataclass_prediction_list
        for im in os.listdir("./dump"):
            im_path = os.path.join(cfg["dump_path"], im)
            prediction = __filter_model(code, cfg["predict"])
            if prediction:
                predict(
                    prediction,
                    cfg["static_dir"],
                    im_path,
                )

    assert True
