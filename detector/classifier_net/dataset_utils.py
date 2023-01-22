import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from detector.classifier_net.config import BATCH_SIZE, IMAGE_SIZE


def get_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


def get_transforms_for_train():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=15, scale=(0.8, 1.3)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])


def folder_to_dataset(path_to_folder_with_images):
    return datasets.ImageFolder(
        root=path_to_folder_with_images,
        transform=get_transforms_for_train()
    )


def dataset_to_loader(dataset):
    return DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=1, pin_memory=True
    )
