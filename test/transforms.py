from __future__ import division
import torch
from cogdata.transforms import transforms
import unittest
import math
import random
import numpy as np
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None

try:
    from scipy import stats
except ImportError:
    stats = None

GRACE_HOPPER = 'data/grace_hopper_517x606.jpg'


class TestTransforms(unittest.TestCase):

    def test_to_tensor(self):
        target = [
            {
                "bbox": [10, 20, 100, 50],
                "segmentation": [[10, 20, 10, 70, 110, 70, 110, 10]],
                "area": 5000,
                "id": 1,
                "iscrowd": 1
            },
            {
                "bbox": [10, 20, 100, 50],
                "segmentation": [[10, 20, 10, 70, 110, 70, 110, 10, 110, 10]],
                "area": 5000,
                "id": 1,
                "iscrowd": 1
            }
        ]
        img = Image.open(GRACE_HOPPER)

        _, targ, target_types = transforms.ToTensor()(img, target)
        self.assertEqual(type(targ), torch.Tensor)
        self.assertTrue(torch.all(torch.tensor([[1.0, 10.0, 20.0, 100.0, 50.0]]*2) == targ))

        _, targ, target_types = transforms.ToTensor([transforms.t_f.TargetType.ID,
                                                     transforms.t_f.TargetType.SEGMENTATION,
                                                     transforms.t_f.TargetType.AREA,
                                                     transforms.t_f.TargetType.BBOX])(img, target)
        self.assertTrue(torch.all(torch.FloatTensor(
            [[1,
              8, 10, 20, 10, 70, 110, 70, 110, 10, 0, 0,
              5000,
              10, 20, 100, 50],
             [1,
              10, 10, 20, 10, 70, 110, 70, 110, 10, 110, 10,
              5000,
              10, 20, 100, 50]
             ]) == targ))

    def test_to_pil_image(self):
        target = [
            {
                "bbox": [10, 20, 100, 50],
                "segmentation": [[10, 20, 10, 70, 110, 70, 110, 10]],
                "area": 5000,
                "id": 1,
                "iscrowd": 1
            },
            {
                "bbox": [10, 20, 100, 50],
                "segmentation": [[10, 20, 10, 70, 110, 70, 110, 10, 110, 10]],
                "area": 5000,
                "id": 1,
                "iscrowd": 1
            }
        ]
        img = Image.open(GRACE_HOPPER)
        img_tens, targ, target_types = transforms.ToTensor()(img, target)
        img2, targ, target_types = transforms.ToPILImage()(img_tens, targ, target_types)
        self.assertTrue(all("id" in ann and "bbox" in ann for ann in targ))
        self.assertTrue(all(ann["id"] == ann_gt["id"] and all(ann["bbox"] == torch.FloatTensor(ann_gt["bbox"]))
                            for ann, ann_gt in zip(targ, target)))

        img_tens, targ, target_types = transforms.ToTensor([transforms.t_f.TargetType.SEGMENTATION,
                                                            transforms.t_f.TargetType.BBOX])(img, target)
        img2, targ, target_types = transforms.ToPILImage()(img_tens, targ, target_types)
        self.assertTrue(all("segmentation" in ann and "bbox" in ann for ann in targ))
        self.assertTrue(all(all(ann["segmentation"] == torch.FloatTensor(ann_gt["segmentation"][0])) and
                            all(ann["bbox"] == torch.FloatTensor(ann_gt["bbox"]))
                            for ann, ann_gt in zip(targ, target)))

    def test_resize(self):

        img = Image.open(GRACE_HOPPER)
        img = img.resize((300, 150))
        img_w, img_h = img.size
        target = [
            {
                "bbox": [0, 0, img_w, img_h],
                "segmentation": [[0, 0, 0, img_h, img_w, img_h, img_w, 0]],
                "area": 5000,
                "id": 1,
                "iscrowd": 1
            },
            {
                "bbox": [0, 0, img_w, img_h],
                "segmentation": [[0, 0, 0, img_h, img_w, img_h, img_w, 0]],
                "area": 5000,
                "id": 1,
                "iscrowd": 1
            }
        ]
        img_trans, targ, target_types = transforms.Resize(100)(img, target)
        img_w, img_h = img_trans.size
        self.assertTrue(all(all(torch.FloatTensor([0, 0, img_w, img_h]) == ann['bbox']) for ann in targ))
        self.assertTrue(all(all(
            torch.FloatTensor([0, 0, 0, img_h, img_w, img_h, img_w, 0]) == ann['segmentation']) for ann in targ))

        img = img.resize((150, 300))
        img_w, img_h = img.size
        target = [
            {
                "bbox": [0, 0, img_w, img_h],
                "segmentation": [[0, 0, 0, img_h, img_w, img_h, img_w, 0]],
                "area": 5000,
                "id": 1,
                "iscrowd": 1
            },
            {
                "bbox": [0, 0, img_w, img_h],
                "segmentation": [[0, 0, 0, img_h, img_w, img_h, img_w, 0]],
                "area": 5000,
                "id": 1,
                "iscrowd": 1
            }
        ]
        img_trans, targ, target_types = transforms.Resize(100)(img, target)
        img_w, img_h = img_trans.size
        self.assertTrue(all(all(torch.FloatTensor([0, 0, img_w, img_h]) == ann['bbox']) for ann in targ))
        self.assertTrue(all(all(
            torch.FloatTensor([0, 0, 0, img_h, img_w, img_h, img_w, 0]) == ann['segmentation']) for ann in targ))

        img_w, img_h = img.size
        target = [
            {
                "bbox": [0, 0, img_w, img_h],
                "segmentation": [[0, 0, 0, img_h, img_w, img_h, img_w, 0]],
                "area": 5000,
                "id": 1,
                "iscrowd": 1
            },
            {
                "bbox": [0, 0, img_w, img_h],
                "segmentation": [[0, 0, 0, img_h, img_w, img_h, img_w, 0]],
                "area": 5000,
                "id": 1,
                "iscrowd": 1
            }
        ]
        img_trans, targ, target_types = transforms.Resize((50, 1050))(img, target)
        img_w, img_h = img_trans.size
        self.assertTrue(all(all(torch.FloatTensor([0, 0, img_w, img_h]) == ann['bbox']) for ann in targ))
        self.assertTrue(all(all(
            torch.FloatTensor([0, 0, 0, img_h, img_w, img_h, img_w, 0]) == ann['segmentation']) for ann in targ))

    def test_random_horizontal_flip(self):
        img = Image.open(GRACE_HOPPER)
        img = img.resize((200, 100))
        target = [
            {
                "bbox": [15, 25, 10, 20],
                "segmentation": [[15, 25, 25, 25, 25, 45, 15, 45]],
                "area": 5000,
                "id": 1,
                "iscrowd": 1
            },
            {
                "bbox": [15, 25, 10, 20],
                "segmentation": [[15, 25, 25, 25, 25, 45, 15, 45, 15, 45]],
                "area": 5000,
                "id": 1,
                "iscrowd": 1
            }
            ]

        img_trans, targ, target_types = transforms.RandomHorizontalFlip(p=1)(img, target)
        self.assertTrue(all(all(torch.FloatTensor([175, 25, 10, 20]) == ann['bbox']) for ann in targ))
        self.assertTrue(all(all(torch.FloatTensor(ann_gt) == ann['segmentation'])
                            for ann, ann_gt in zip(targ, [[185, 25, 175, 25, 175, 45, 185, 45],
                                                          [185, 25, 175, 25, 175, 45, 185, 45, 185, 45]])))

    def test_random_resize_pad(self):
        img = Image.open(GRACE_HOPPER)
        img_w, img_h = img.size
        target = [
            {
                "bbox": [0, 0, img_w, img_h],
                "segmentation": [[0, 0, 0, img_h, img_w, img_h, img_w, 0]],
                "area": 5000,
                "id": 1,
                "iscrowd": 1
            },
            {
                "bbox": [15, 25, 10, 20],
                "segmentation": [[15, 25, 25, 25, 25, 45, 15, 45, 15, 45]],
                "area": 5000,
                "id": 1,
                "iscrowd": 1
            }
        ]

        for i in range(100):
            test_img = img.resize((random.randint(100, 1000), random.randint(100, 1000)))
            min_x = random.randint(100, 1000)
            min_y = random.randint(100, 1000)
            max_x = random.randint(min_x, 1000)
            max_y = random.randint(min_y, 1000)
            img_trans, targ, target_types = transforms.RandomResizePad(min_size=(min_x, min_y),
                                                                       max_size=(max_x, max_y))(img, target)
            self.assertTrue(img_trans.size[1] == max_x and img_trans.size[0] == max_y)
            # img_trans.show()
