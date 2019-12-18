import typing
from enum import Enum
import torch
import sys
import collections
if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


class TargetType(Enum):
    BBOX = 'bbox'
    SEGMENTATION = 'segmentation'
    AREA = 'area'
    ID = 'id'
    IS_CROWD = 'iscrowd'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_


TargetTypes = typing.Dict[TargetType, typing.Tuple[int, int]]


def get_ann_value(ann: torch.Tensor, target_type: TargetType, target_types: TargetTypes):
    start_i, end_i = target_types[target_type]
    t = ann[start_i: end_i]
    if target_type == TargetType.SEGMENTATION:
        t = t[1: int(t[0])+1]
    if len(t) == 0:
        t = t[0]
    return t


def to_tensor(target: dict, target_types: typing.List[TargetType], max_segmentation_length=0):
    """
    Convert dictionary of targets to a tensor of targets.
    :param target: Dictionary containing targets
    :param target_types: List of targets types to return
    :param max_segmentation_length: Largest number of points in the segmentation
    :return: tensor of targets
    """
    out_t = []

    if TargetType.SEGMENTATION in target_types:
        max_hold = max(len(ann[TargetType.SEGMENTATION.value][0]) for ann in target)
        max_segmentation_length = max(max_segmentation_length, max_hold)
    res_target_types = {}
    for ann in target:
        out_ann = []
        for target_type in target_types:
            val = ann[target_type.value]
            if target_type == TargetType.SEGMENTATION:
                assert len(val) == 1 and len(val[0]) % 2 == 0
                val = [len(val[0])] + val[0] + [0]*(max_segmentation_length - len(val[0]))
            elif target_type == TargetType.BBOX:
                assert len(val) == 4
            else:
                val = [val]
            start_i = len(out_ann)
            out_ann.extend(val)
            res_target_types[target_type] = (start_i, len(out_ann))  # Smells a bit
        out_t.append(out_ann)
    return torch.FloatTensor(out_t), res_target_types


def to_dict(target: torch.Tensor, target_types: TargetTypes) -> dict:
    """
    Convert targets to dictionary form
    :param target: Tensor containing the target
    :param target_types: List of target types contained in the target
    :return: dict of the various target types
    """
    res_target = []
    for ann in target:
        res_ann = {}
        for target_type in target_types.keys():
            res_ann[target_type.value] = get_ann_value(ann, target_type, target_types)
        res_target.append(res_ann)
    return res_target


def apply_to_target(target: typing.Union[torch.Tensor, typing.List[dict]], target_types: TargetTypes,
                    f: typing.Callable, **kwargs):
    """
    Maps to target to the respective function
    :param target:
    :param target_types:
    :param f: Function to call
    :param kwargs:
    :return:
    """
    if isinstance(target, list) and len(target) > 0 and isinstance(target[0], dict):
        for ann in target:
            for key, value in ann.items():
                if not TargetType.has_value(key):
                    continue
                if isinstance(value, Iterable):
                    value = torch.FloatTensor(value).flatten()[None]
                    ann[key] = f(value, TargetType(key), **kwargs)[0]
                else:
                    ann[key] = f(value, TargetType(key), **kwargs)
        return target
    elif isinstance(target, torch.Tensor):
        for target_type, (start_i, end_i) in target_types.items():
            target[:, start_i: end_i] = f(target[:, start_i: end_i], target_type, **kwargs)
    elif isinstance(target, list) and len(target) <= 0:
        pass
    else:
        raise ValueError("Target must be of type dict or Tensor.")
    return target


def resize(target: torch.Tensor, target_type: TargetType,
           orig_size: typing.Tuple[int, int], new_size: typing.Tuple[int, int]) -> torch.Tensor:
    if target_type == TargetType.SEGMENTATION or target_type == TargetType.BBOX:
        scale_x, scale_y = float(new_size[0])/orig_size[0], float(new_size[1])/orig_size[1]
        target[:, ::2] *= scale_x
        target[:, 1::2] *= scale_y
    return target


def pad(target: torch.Tensor, target_type: TargetType, top: int, left: int) -> torch.Tensor:
    if target_type == TargetType.SEGMENTATION:
        target[:, ::2] += left
        target[:, 1::2] += top
    elif target_type == TargetType.BBOX:
        target[:, 0] += left
        target[:, 1] += top

    return target


def hflip(target: torch.Tensor, target_type: TargetType, img_size: typing.Tuple[int, int]) -> torch.Tensor:
    if target_type == TargetType.BBOX:
        target[:, 0] = img_size[0] - target[:, 0] - target[:, 2]
    elif target_type == TargetType.SEGMENTATION:
        target[:, ::2] = img_size[0] - target[:, ::2]
    return target
