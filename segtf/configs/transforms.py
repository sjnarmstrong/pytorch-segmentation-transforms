from typing_extensions import Literal
from pydantic import BaseModel, Extra
from typing import Union, List, Tuple
from segtf.configs.augdata import InterpolationMethods
from enum import Enum


class TargetType(Enum):
    BBOX = 'bbox'
    SEGMENTATION = 'segmentation'
    AREA = 'area'
    ID = 'id'
    IS_CROWD = 'iscrowd'


class Compose(BaseModel):
    class Config:
        title = 'Compose'
        description = ''
    ID: Literal['Compose']
    transforms: List['TransformsType']

    def __call__(self):
        from cogdata.transforms.transforms import Compose
        return Compose(self.transforms)


class ToTensor(BaseModel):
    class Config:
        title = 'ToTensor'
        description = ''
        extra = Extra.forbid
    ID: Literal['ToTensor']
    target_types: List[TargetType] = [TargetType.BBOX, TargetType.ID]

    def __call__(self):
        from cogdata.transforms.transforms import ToTensor
        return ToTensor()


class ToPILImage(BaseModel):
    class Config:
        title = 'ToPILImage'
        description = ''
        extra = Extra.forbid
    ID: Literal['ToPILImage']

    def __call__(self):
        from cogdata.transforms.transforms import ToPILImage
        return ToPILImage()


class Normalize(BaseModel):
    class Config:
        title = 'Normalize'
        description = ''
        extra = Extra.forbid
    ID: Literal['Normalize']
    mean: Union[tuple, list, int, float] = [0.485, 0.456, 0.406]
    std: Union[tuple, list, int, float] = [0.229, 0.224, 0.225]
    inplace: bool = False

    def __call__(self):
        from cogdata.transforms.transforms import Normalize
        return Normalize(mean=self.mean, std=self.std, inplace=self.inplace)


class Resize(BaseModel):
    class Config:
        title = 'Resize'
        description = ''
        extra = Extra.forbid
    ID: Literal['Resize']

    size: Union[tuple, list, int, float]
    interpolation: InterpolationMethods = InterpolationMethods.LINEAR

    def __call__(self):
        from cogdata.transforms.transforms import Resize
        return Resize(size=self.size, interpolation=self.interpolation.value)


class RandomHorizontalFlip(BaseModel):
    class Config:
        title = 'RandomHorizontalFlip'
        description = 'RandomHorizontalFlip the data'
        extra = Extra.forbid
    ID: Literal['RandomHorizontalFlip']

    p: float = 0.5

    def __call__(self):
        from cogdata.transforms.transforms import RandomHorizontalFlip
        return RandomHorizontalFlip(p=self.p)


class RandomResizePad(BaseModel):
    class Config:
        title = 'RandomResizePad'
        description = 'RandomResizePad the data'
        extra = Extra.forbid
    ID: Literal['RandomResizePad']

    min_size: Tuple[int, int]
    max_size: Tuple[int, int]
    diff_aspect_ratio: Tuple[int, int] = (0.95, 1.05)
    fill: Union[None, str, Tuple[int, int, int], float, int] = None
    interpolation: InterpolationMethods = InterpolationMethods.BILINEAR

    def __call__(self):
        from cogdata.transforms.transforms import RandomResizePad
        return RandomResizePad(min_size=self.min_size, max_size=self.max_size, diff_aspect_ratio=self.diff_aspect_ratio,
                               fill=self.fill, interpolation=self.interpolation.value)


TransformsType = Union[Compose, ToTensor, ToPILImage, Normalize, Resize, RandomHorizontalFlip, RandomResizePad]

Compose.update_forward_refs()
