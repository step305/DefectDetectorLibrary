import json
import base64
import pathlib

import cv2
import numpy
import torch
import torch.utils.data as torch_data
import detector.detector_net.transforms as T

from PIL import Image
# reworked from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


def _create_masks(shapes, image_width: int, image_height: int):
    for shape in shapes:
        mask = numpy.zeros((image_height, image_width), dtype=numpy.uint8)
        points = numpy.array(shape['points']).reshape((-1, 1, 2))
        points = numpy.round(points).astype(numpy.int32)

        cv2.fillPoly(mask, [points], (1, ))
        mask = mask.astype(numpy.uint8)
        yield mask


def _create_bboxs(shapes):
    for shape in shapes:
        points = numpy.array(shape['points'])
        xmin, ymin = numpy.min(points, axis=0)
        xmax, ymax = numpy.max(points, axis=0)

        yield [xmin, ymin, xmax, ymax]


class LabelMeDataset(torch_data.Dataset):
    def __init__(self, directory, transforms):
        self.directory = pathlib.Path(directory)
        assert self.directory.exists()
        assert self.directory.is_dir()

        self.labelme_paths = []

        for labelme_path in self.directory.rglob('*.json'):
            with open(labelme_path, 'r') as labelme_file:
                labelme_json = json.load(labelme_file)

                required_keys = ['version', 'flags', 'shapes', 'imagePath', 'imageData', 'imageHeight', 'imageWidth']
                assert all(key in labelme_json for key in required_keys), (required_keys, labelme_json.keys())

                self.labelme_paths += [labelme_path]

        self.transforms = transforms

    def __len__(self):
        return len(self.labelme_paths)

    def __getitem__(self, idx: int):
        labelme_path = self.labelme_paths[idx]
        image_path = str(labelme_path).replace('json', 'jpg')

        with open(labelme_path, 'r') as labelme_file:
            labelme_json = json.load(labelme_file)

        image_width = labelme_json['imageWidth']
        image_height = labelme_json['imageHeight']

        image = Image.open(image_path).convert('RGB')

        labelme_shapes = [i for i in labelme_json['shapes'] if len(i['points']) > 2]
        assert all(i['shape_type'] == 'polygon' for i in labelme_shapes)

        masks = list(_create_masks(labelme_shapes, image_width, image_height))
        bboxes = list(_create_bboxs(labelme_shapes))

        target = {}
        target['masks'] = torch.as_tensor(numpy.stack(masks), dtype=torch.uint8)
        target['labels'] = torch.ones((len(labelme_shapes),), dtype=torch.int64)
        target['iscrowd'] = torch.zeros_like(target['labels'], dtype=torch.int64)
        target['image_id'] = torch.tensor([idx], dtype=torch.int64)

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)

        target['area'] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target['boxes'] = bboxes
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == '__main__':
    dat = LabelMeDataset('..\\..\\dataset\\detection_train', get_transform(True))
    print(dat.__len__())
    img, tar = dat.__getitem__(1)
    print(tar)
    print(img.shape)
    print(tar['masks'].sum())
