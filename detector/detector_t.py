import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import cv2
import torchvision.models.segmentation
import torch
from detector.detector_net.dataset import get_transform
from PIL import Image
import detector.detector_net.dataset as labelme_dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import time

IMAGE_SIZE = (600, 600)


class Detector:
    def __init__(self, model_path):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                                hidden_layer,
                                                                2)
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        self.transforms = get_transform(False)

    def test(self, image):
        old_height, old_width, _ = image.shape
        image = cv2.resize(image, IMAGE_SIZE, cv2.INTER_LINEAR)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        if self.transforms is not None:
            image_infer, _ = self.transforms(im_pil, {})
        bboxes, scores = self.infer(image_infer)
        k_width = old_width / IMAGE_SIZE[0]
        k_height = old_height / IMAGE_SIZE[1]
        new_bboxes = []
        for bbox in bboxes:
            x1_, y1_, x2_, y2_ = bbox
            new_bboxes.append(
                (
                    int(x1_ * k_width),
                    int(y1_ * k_height),
                    int(x2_ * k_width),
                    int(y2_ * k_height)
                )
            )
        return new_bboxes, scores

    def infer(self, image):
        images = [image.to(self.device)]

        with torch.no_grad():
            pred = self.model(images)
        bboxes = []
        scores = []
        for i in range(len(pred[0]['labels'])):
            score = pred[0]['scores'][i].detach().cpu().numpy()
            if score > 0.4:
                scores.append(float(score))
                bbox = pred[0]['boxes'][i].detach().cpu().numpy()
                b = list(map(int, bbox.tolist()))
                bboxes.append(b)
        return bboxes, scores

    def verification(self):
        dataset = labelme_dataset.LabelMeDataset('dataset\\detection_verification', get_transform(False))
        infer_times = []
        image, target = dataset.__getitem__(0)
        _, _ = self.infer(image)
        metric = MeanAveragePrecision()
        for i in range(dataset.__len__()):
            image, target = dataset.__getitem__(i)
            t0 = time.time()
            boxes, scores = self.infer(image)
            t1 = time.time()
            infer_times.append((t1 - t0) * 1000.0)
            target_metric = [
                dict(
                    boxes=target['boxes'],
                    labels=target['labels']
                )
            ]
            pred_metrics = [
                dict(
                    boxes=torch.tensor(boxes),
                    scores=torch.tensor(scores),
                    labels=torch.tensor([1] * len(scores))
                )
            ]
            metric.update(pred_metrics, target_metric)

        res_m = metric.compute()
        mAP = float(res_m['map_50']) * 100.0
        verification_result = {
            'total_images': len(infer_times),
            'mAP50': mAP,
            'average_t': sum(infer_times) / len(infer_times),
            'max_t': max(infer_times),
            'std_t': np.array(infer_times).std(),
        }
        return verification_result


if __name__ == '__main__':
    det = Detector('models\\rcnn\\5.torch')
    img_test = cv2.imread('test_rcnn.jpg')
    bb, sc = det.test(img_test)
    for coord, score in zip(bb, sc):
        x1, y1, x2, y2 = coord
        cv2.rectangle(img_test,
                      (x1, y1),
                      (x2, y2),
                      color=(0, 0, 255),
                      thickness=2)
        cv2.putText(img_test,
                    '{:.2f}% {:}'.format(score * 100.0, score),
                    (x1, y1 - 10),
                    fontFace=cv2.FONT_ITALIC,
                    fontScale=0.4,
                    thickness=1,
                    color=(255, 0, 0))
    cv2.imwrite('test_rcnn_detected.jpg', img_test)

    res = det.verification()
    print(res)
