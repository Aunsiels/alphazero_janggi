import torch
from torch import nn

from janggi.utils import BOARD_HEIGHT, BOARD_WIDTH

ACTION_SIZE = 58
IN_CHANNEL = 16
OUT_CHANNEL = 256
STRIDE_SIZE = 3
OUT_CONV_POLICY = 2
OUT_LINEAR_VALUE = 256

N_RESIDUAL_DEFAULT = 2


class JanggiNetwork(nn.Module):

    def __init__(self, n_residual=N_RESIDUAL_DEFAULT):
        super().__init__()

        self.first_layer = FirstLayerJanggiNetwork()
        self.residuals = nn.ModuleList([ResidualBlock() for _ in range(n_residual)])
        self.policy_network = PolicyNetwork()
        self.value_network = ValueNetwork()

    def forward(self, x):
        x = self.first_layer(x)
        for residual in self.residuals:
            x = residual(x)
        policy = self.policy_network(x)
        value = self.value_network(x)
        return policy, value


class ValueNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(OUT_CHANNEL, 1, 1)
        self.batchnorm = nn.BatchNorm2d(1)
        self.first_linear = nn.Linear(BOARD_HEIGHT * BOARD_WIDTH, OUT_LINEAR_VALUE)
        self.relu = nn.ReLU()
        self.second_linear = nn.Linear(OUT_LINEAR_VALUE, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = x.view(-1, BOARD_HEIGHT * BOARD_WIDTH)
        x = self.first_linear(x)
        x = self.relu(x)
        x = self.second_linear(x)
        x = self.tanh(x)
        return x


class PolicyNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(OUT_CHANNEL, OUT_CONV_POLICY, 1)
        self.batchnorm = nn.BatchNorm2d(OUT_CONV_POLICY)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(OUT_CONV_POLICY * BOARD_WIDTH * BOARD_HEIGHT,
                                ACTION_SIZE * BOARD_WIDTH * BOARD_HEIGHT)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = x.view(-1, OUT_CONV_POLICY * BOARD_WIDTH * BOARD_HEIGHT)
        x = self.linear(x)
        x = self.softmax(x)
        x = x.view(-1, ACTION_SIZE, BOARD_HEIGHT, BOARD_WIDTH)
        return x


class FirstLayerJanggiNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # First block:
        self.first_conv = nn.Conv2d(IN_CHANNEL, OUT_CHANNEL, STRIDE_SIZE, padding=1)
        self.first_batchnorm = nn.BatchNorm2d(OUT_CHANNEL)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.first_conv(x)
        x = self.first_batchnorm(x)
        x = self.relu(x)

        return x


class ResidualBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.first_conv = nn.Conv2d(OUT_CHANNEL, OUT_CHANNEL, STRIDE_SIZE, padding=1)
        self.first_batchnorm = nn.BatchNorm2d(OUT_CHANNEL)
        self.first_relu = nn.ReLU()
        self.second_conv = nn.Conv2d(OUT_CHANNEL, OUT_CHANNEL, STRIDE_SIZE, padding=1)
        self.second_batchnorm = nn.BatchNorm2d(OUT_CHANNEL)
        self.second_relu = nn.ReLU()
        self.identity = nn.Identity()

    def forward(self, x):
        x2 = self.identity(x)
        x = self.first_conv(x)
        x = self.first_batchnorm(x)
        x = self.first_relu(x)
        x = self.second_conv(x)
        x = self.second_batchnorm(x)
        x += x2
        x = self.second_relu(x)
        return x


class JanggiLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        policy_found = target[0].view(-1, ACTION_SIZE * BOARD_WIDTH * BOARD_HEIGHT)
        policy_guessed = output[0].view(-1, ACTION_SIZE * BOARD_WIDTH * BOARD_HEIGHT)
        loss_value = (output[1] - target[1]) ** 2
        loss_policy = torch.bmm(policy_found.view(-1, 1, ACTION_SIZE * BOARD_WIDTH * BOARD_HEIGHT),
                                torch.log(policy_guessed).view(-1, ACTION_SIZE * BOARD_WIDTH * BOARD_HEIGHT, 1))
        return (loss_value - loss_policy).view(-1).mean()
