import detector.detector_net.dataset as labelme_dataset
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.models.segmentation
from detector.detector_net.engine import train_one_epoch, evaluate
import detector.detector_net.utils as utils
from detector.detector_net.dataset import get_transform


IMAGE_SIZE = (600, 600)
TORCH_DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Train using {}'.format(TORCH_DEVICE))


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def train_detector():
    dataset = labelme_dataset.LabelMeDataset('dataset\\detection_train', get_transform(True))
    dataset_test = labelme_dataset.LabelMeDataset('dataset\\detection_train', get_transform(False))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)
    model = get_model_instance_segmentation(2)

    model.to(TORCH_DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 301

    for i in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, TORCH_DEVICE, i, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device=TORCH_DEVICE)

        if i % 50 == 0:
            torch.save(model.state_dict(), 'detector\\models\\rcnn\\' + str(i) + ".torch")
            print("Saved model to:", str(i) + ".torch")


if __name__ == '__main__':
    train_detector()
