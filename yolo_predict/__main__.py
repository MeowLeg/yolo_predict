import os
import sqlite3
from time import sleep
from typing import Any

import requests
import tomli
from ultralytics.models.yolo.model import YOLO

from yolo_predict.utils.db import read_images_from_db
from yolo_predict.utils.detect import (
    Alert,
    Image,
    Prediction,
    predict,
)


def __filter_model(code: str, cfg_predicts: list[Prediction]) -> Prediction | None:
    # 列出实现的模型
    for p in cfg_predicts:
        if p.tag == code:
            return p


def __update_image_status(cur: sqlite3.Cursor, image_id: int) -> None:
    _ = cur.execute("update pic set predicted = 1 where id = ?", (image_id,))


def __path_2_url(url_base: str, im_path: str) -> str:
    return f"{url_base}/{os.path.basename(im_path)}"


def loop_predict(
    cfg: dict[str, Any],
) -> None:
    # 循环检测
    db = sqlite3.connect(cfg["db_path"])
    db.row_factory = sqlite3.Row
    cur = db.cursor()
    while True:
        im_predicted: list[Alert] = []
        predicted_image_id: int | None = None
        im: Image | None = None
        for im in read_images_from_db(cur, cfg["read_pic_limit"]):
            if predicted_image_id is None:
                predicted_image_id = im.id
            elif predicted_image_id != im.id:
                __update_image_status(cur, predicted_image_id)
                predicted_image_id = im.id
            prediction = __filter_model(im.code, cfg["predicts"])
            if prediction and predict(
                prediction,
                cfg["static_path"],
                im.path,
            ):
                im_predicted.append(
                    Alert(
                        im.uuid,
                        "",
                        "",
                        im.create_date,
                        [prediction.tag],
                        __path_2_url(cfg["static_base_url"], im.path),
                        [],
                        [],
                    )
                )
            # todo: 复合检测的情况
        if im:
            # 这里很有可能把在批处理间隔开的额同一张图片给提前udpate了
            # 所以在db模块里对数据做了一些冗余的处理步骤
            __update_image_status(cur, im.id)
        db.commit()
        # 通知远程服务器，todo里的yoloAlert.json
        for p in im_predicted:
            # 正式环境 https://fh2.wifizs.cn/11005/v1/webhook/yoloalert
            response = requests.post(cfg["notify_url"], json=p)
            print(response.text)
        # 等待
        sleep(cfg["watch_interval"])


def __load_config():
    with open("./config.toml", "rb") as f:
        cfg: dict[str, Any] = tomli.load(f)  #  pyright: ignore[reportAny]
        dataclass_prediction_list = []
        for p in cfg["predict"]:
            p["yolo"] = YOLO(p["model"])
            dataclass_prediction_list.append(Prediction(**p))
        cfg["predict"] = dataclass_prediction_list
        return cfg


def main():
    loop_predict(__load_config())


if __name__ == "__main__":
    main()
