# Pytorch Segmentation Transforms

> Transforms for segmentation based data (Multiple labels for a single image)

This repository is inspired by the transformations implemented in torchvision. However, no current method exists to transform both the target and the image at the same time. For example, when resizing an image, torchvision does not provide a method to resize the bounding boxes. The code here is still under development and many of the functions do not work. It is assumed that the label data has the following format (taken from the COCO dataset):

    {
        "bbox": [x, y, w, h],
        "segmentation": [[x1, y1, x2, y2, x3, y3, x4, y4]],
        "area": w*h,
        "id": 1,
        "iscrowd": 1
    },

The ToTensor transform also transforms this json format into a flat tensor format as follows:
[x, y, w, h, 4, x1, y1, x2, y2, x3, y3, x4, y4, w*h, 1, 1]
The order of which is specified to the function. 

---

## Installation

It is possible to install this package directly from this git page with the use of pip. This package has not yet been deployed to pypi as it is still in the early stages of development

## Usage

- Coming soon

---

## Contribution

Please feel free to contribute to this project. The more hands the better. This can be done with a pull request. I have many projects that I am currently working on so i may not always be able to assist.

## License

MIT
