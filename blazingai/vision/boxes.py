import json
from typing import Any

import numpy as np
import pandas as pd
import torch

JsonDict = dict[str, Any]


def boxes_str2array(bboxes_str: list[str]) -> list:
    boxes_json = _boxes_str2json(bboxes_str)
    boxes_array = _boxes_json2array(boxes_json)
    return boxes_array


def _boxes_str2json(bboxes_str: list[str]) -> list[JsonDict]:
    result = [
        json.loads(idx.replace("'", '"')) if idx is not np.nan else np.nan
        for idx in bboxes_str
    ]
    return result


def _boxes_json2array(
    bboxes_json: list[JsonDict], fillna: list = [[0, 0, 1, 1]]
) -> list:
    result = []
    for target in bboxes_json:
        if target is not np.nan:
            value = pd.DataFrame(target).values.reshape(-1, 4)
        else:
            value = np.array(fillna).reshape(-1, 4)
        result.append(value)
    return result


def _boxes_coco2pascalvoc(boxes: torch.Tensor) -> torch.Tensor:
    """[x_min, y_min, width, height] -> [x_min, y_min, x_max, y_max]"""
    boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
    return boxes


def _boxes_pascalvoc2coco(boxes: torch.Tensor) -> torch.Tensor:
    """[x_min, y_min, x_max, y_max] -> [x_min, y_min, width, height]"""
    boxes[:, 2:] = boxes[:, 2:] - boxes[:, :2]
    return boxes


convert_boxes = {
    "coco2pascalvoc": _boxes_coco2pascalvoc,
    "pascalvoc2coco": _boxes_pascalvoc2coco,
}
