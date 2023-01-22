import torch
from CNNModel import CNNModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_model(path_to_model, epochs, model, optimizer, criterion):
    torch.save(
        {
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        },
        path_to_model
    )


def load_model(target_device, path_to_model):
    model = CNNModel().to(target_device)
    checkpoint = torch.load(path_to_model, map_location=target_device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def save_plot(title, train_metric, validation_metric):
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_metric, color='green', linestyle='-',
        label='train'
    )
    plt.plot(
        validation_metric, color='blue', linestyle='-',
        label='validation'
    )
    plt.xlabel('Epoch count')
    plt.ylabel(title)
    plt.legend()
    plt.savefig(title + '.png')
