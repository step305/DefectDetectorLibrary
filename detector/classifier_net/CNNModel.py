import torch.nn as nn
import torch.nn.functional as nn_functional


class CNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNModel, self).__init__()
        # convolution layer #1
        # input size [batch size x 3 x 64 x 64]
        self.convolution_1 = nn.Conv2d(in_channels=3,
                                       out_channels=18,
                                       kernel_size=5)

        # convolution layer #2
        # input size [batch size x 18 x 60 x 60]
        self.convolution_2 = nn.Conv2d(in_channels=18,
                                       out_channels=16,
                                       kernel_size=5)

        # linear transformation layer #1
        # input size [batch size x 16 x 13 x 13]
        self.linear_transform_1 = nn.Linear(16 * 13 * 13, 120)

        # linear transformation layer #2
        # input size [batch size x 120]
        self.linear_transform_2 = nn.Linear(120, 84)

        # linear transformation layer #3
        # input size [batch size x 84]
        self.linear_transform_3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # convolution layer #1
        x = self.convolution_1(x)
        # 2D max pooling over input signals
        x = nn_functional.max_pool2d(x, kernel_size=2)
        # rectified linear unit function over input signals
        x = nn_functional.relu(x)
        # convolution layer #2
        x = self.convolution_2(x)
        # 2D max pooling over input signals
        x = nn_functional.max_pool2d(x, kernel_size=2)
        # rectified linear unit function over input signals
        x = nn_functional.relu(x)
        # reshape data
        x = x.view(x.shape[0], -1)
        # linear transformation layer #1
        x = self.linear_transform_1(x)
        # rectified linear unit function over input signals
        x = nn_functional.relu(x)
        # linear transformation layer #2
        x = self.linear_transform_2(x)
        # rectified linear unit function over input signals
        x = nn_functional.relu(x)
        # x = nn_functional.softmax(x, dim=1)
        # linear transformation layer #3
        out = self.linear_transform_3(x)
        # out - array of probabilities of "x" is in target classes
        return out
