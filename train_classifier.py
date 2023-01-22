from detector.classifier_net.config import *
import detector.classifier_net.dataset_utils as datasets
import contextlib
import time
import torch
import torch.nn as nn
import torch.optim as optim
from detector.classifier_net.CNNModel import CNNModel
from detector.classifier_net import utils

TARGET_DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')


def train(cnn_model, data_loader, train_criterion, train_optimizer=None, is_validation=False):
    if is_validation:
        print('Start validation process...')
        cnn_model.eval()
    else:
        print('Start training process...')
        cnn_model.train()

    loss = 0.0
    correct = 0
    counter = 0
    performance_time = time.time()

    with torch.no_grad() if is_validation else contextlib.suppress():
        # doesn't need to calc gradients when validation is in progress
        for data in data_loader:
            counter += 1
            image, labels = data
            image = image.to(TARGET_DEVICE)
            labels = labels.to(TARGET_DEVICE)
            if not is_validation:
                # clear gradients on optimizer
                train_optimizer.zero_grad()
            # do forward pass
            outputs = cnn_model(image)
            # calculate the loss between estimation and target
            loss_ = train_criterion(outputs, labels)
            # calculate the accuracy
            _, predictions = torch.max(outputs.data, 1)
            correct += sum([1 if prediction == label else 0 for prediction, label in zip(predictions, labels)])
            loss += sum([0 if prediction == label else 1 for prediction, label in zip(predictions, labels)])
            if not is_validation:
                # do loss backpropagation
                loss_.backward()
                # update the optimizer
                train_optimizer.step()
    performance_time = time.time() - performance_time

    # loss and accuracy for the epoch
    epoch_loss = 100.0 * float(loss) / len(data_loader.dataset)
    epoch_acc = 100.0 * float(correct) / len(data_loader.dataset)
    return epoch_loss, epoch_acc, performance_time


if __name__ == '__main__':
    print('Model will be trained using: {}'.format(TARGET_DEVICE.capitalize()))
    model = CNNModel().to(TARGET_DEVICE)
    TOTAL_MODEL_PARAMETERS = sum(p.numel() for p in model.parameters())
    print('Total model parameters: {}'.format(TOTAL_MODEL_PARAMETERS))
    TOTAL_TRAINABLE_PARAMETERS = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print('Number of parameters to train:{}'.format(TOTAL_TRAINABLE_PARAMETERS))

    # Adam Method for Stochastic Optimization.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #  loss function for training - cross entropy loss between estimation and target
    criterion = nn.CrossEntropyLoss()

    train_loss_history = []
    validation_loss_history = []
    train_accuracy_history = []
    validation_accuracy_history = []

    # prepare datasets
    train_dataset = datasets.folder_to_dataset('dataset/classification/train')
    validation_dataset = datasets.folder_to_dataset('dataset/classification/validation')
    train_loader = datasets.dataset_to_loader(train_dataset)
    validation_loader = datasets.dataset_to_loader(validation_dataset)

    performance = 0.0

    # train model
    for epoch in range(TRAIN_EPOCHS):
        print(f"[INFO]: Epoch {epoch + 1} of {TRAIN_EPOCHS}")
        train_epoch_loss, train_epoch_acc, train_timer = train(model, train_loader,
                                                               criterion, optimizer, is_validation=False)
        validation_epoch_loss, validation_epoch_acc, validation_timer = train(model, validation_loader,
                                                                              criterion, is_validation=True)
        performance += train_timer + validation_timer
        print('[INFO]: Epoch {} of {} done: train in {:.1f}ms, validation in {:.1f}ms'.format(epoch + 1,
                                                                                              TRAIN_EPOCHS,
                                                                                              train_timer * 1000.0,
                                                                                              validation_timer * 1000.0)
              )
        print('Training loss: {:.3f}, training acc: {:.3f}'.format(train_epoch_loss, train_epoch_acc))
        print('Validation loss: {:.3f}, validation acc: {:.3f}'.format(validation_epoch_loss, validation_epoch_acc))
        print('-' * 60)
        train_loss_history.append(train_epoch_loss)
        validation_loss_history.append(validation_epoch_loss)
        train_accuracy_history.append(train_epoch_acc)
        validation_accuracy_history.append(validation_epoch_acc)

    # save trained model
    utils.save_model('detector\\models\\classification_model.pth', TRAIN_EPOCHS, model, optimizer, criterion)
    # save loss and accuracy plots
    utils.save_plot('Accuracy', train_accuracy_history, validation_accuracy_history)
    utils.save_plot('Loss', train_loss_history, validation_loss_history)
    print('Training complete!')
    print('Train performance: {:.2f}fps'.format(
        (len(train_loader.dataset) + len(validation_loader.dataset)) * TRAIN_EPOCHS / performance)
    )
    print('Model saved to classification_model.pth')
