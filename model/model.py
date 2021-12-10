import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict


class TransferedLearned(nn.Module):
    def __init__(self):
        super(TransferedLearned, self).__init__()
        # load pretrained model for feature extraction
        self.feature_extractor = torchvision.models.resnet18(pretrained=True)
        # freeze feature extractor part
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # replace last layer with an indentity layer (to remove the last fc layer)
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Identity(num_ftrs)

        # embedding layer for position from sites inputs
        embedding_dim = 12
        self.pos_embedding = nn.Embedding(
            num_embeddings=12, embedding_dim=embedding_dim
        )

        # add new fc layers
        self.fc1 = nn.Linear(num_ftrs + embedding_dim, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, images, sites, masks):
        # images is a tensor of batch_size x num_sites x 3 x 224 x 224

        feature_vectors = []
        for i in range(images.shape[1]):
            each_image_site = images[:, i, :, :, :]
            each_image_site = each_image_site.view(
                images.shape[0], images.shape[2], images.shape[3], images.shape[4]
            )

            # x is now batch_size x 3 x 224 x 224
            x = self.feature_extractor(each_image_site)
            feature_vectors.append(x)

        # position embedding of size batch_size x num_sites x embedding_dim
        embedded_sites = self.pos_embedding(sites)

        # stack all feature vectors to a new dimension of size batch_size x num_sites x num_features (512 for ResNet18)
        x = torch.stack(feature_vectors, dim=1)

        # concatenate feature vectors and position embeddings (batch_size x num_sites x [num_features + embedding_dim])
        x = torch.cat([x, embedded_sites], dim=2)

        masks = masks == 1  # convert to boolean
        masks = masks.unsqueeze(-1).expand(
            x.size()
        )  # expand from batch_size x num_sites to batch_size x num_sites x num_features (512 for ResNet18)

        x = x * masks  # apply masks and preserver the original tensor dimensions

        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))

        # average all feature vectors
        x = torch.mean(x, dim=1)  # batch_size x [num_features + embedding_dim]

        # x = self.dropout(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


# paper: https://www.nature.com/articles/s41598-019-42557-4
# code from: https://github.com/FrancescoSaverioZuppichini/ResNet
# Basic Block - Conv2d with auto padding
class Conv2dAuto(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (
            self.kernel_size[0] // 2,
            self.kernel_size[1] // 2,
        )  # dynamic add padding based on the kernel_size


# Residual Block -
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.blocks = nn.Identity()
        self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        if self.should_apply_shortcut:
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return x

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class ResNetResidualBlock(ResidualBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=1,
        downsampling=1,
        conv=partial(Conv2dAuto, kernel_size=3, bias=False),
        *args,
        **kwargs
    ):
        super().__init__(in_channels, out_channels)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        self.shortcut = (
            nn.Sequential(
                OrderedDict(
                    {
                        "conv": nn.Conv2d(
                            self.in_channels,
                            self.expanded_channels,
                            kernel_size=1,
                            stride=self.downsampling,
                            bias=False,
                        ),
                        "bn": nn.BatchNorm2d(self.expanded_channels),
                    }
                )
            )
            if self.should_apply_shortcut
            else None
        )

    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels


def conv_bn(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(
        OrderedDict(
            {
                "conv": conv(in_channels, out_channels, *args, **kwargs),
                "bn": nn.BatchNorm2d(out_channels),
            }
        )
    )


class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(
                self.in_channels,
                self.out_channels,
                conv=self.conv,
                bias=False,
                stride=self.downsampling,
            ),
            activation(),
            conv_bn(
                self.out_channels, self.expanded_channels, conv=self.conv, bias=False
            ),
        )


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4

    def __init__(self, in_channels, out_channels, activation=nn.ReLU, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
            conv_bn(self.in_channels, self.out_channels, self.conv, kernel_size=1),
            activation(),
            conv_bn(
                self.out_channels,
                self.out_channels,
                self.conv,
                kernel_size=3,
                stride=self.downsampling,
            ),
            activation(),
            conv_bn(
                self.out_channels, self.expanded_channels, self.conv, kernel_size=1
            ),
        )


class ResNetLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs
    ):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1

        self.blocks = nn.Sequential(
            block(
                in_channels, out_channels, *args, **kwargs, downsampling=downsampling
            ),
            *[
                block(
                    out_channels * block.expansion,
                    out_channels,
                    downsampling=1,
                    *args,
                    **kwargs
                )
                for _ in range(n - 1)
            ]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by increasing different layers with increasing features.
    """

    def __init__(
        self,
        in_channels=3,
        blocks_sizes=[64, 128, 256, 512],
        deepths=[2, 2, 2, 2],
        activation=nn.ReLU,
        block=ResNetBasicBlock,
        *args,
        **kwargs
    ):
        super().__init__()

        self.blocks_sizes = blocks_sizes

        self.gate = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.blocks_sizes[0],
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activation(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList(
            [
                ResNetLayer(
                    blocks_sizes[0],
                    blocks_sizes[0],
                    n=deepths[0],
                    activation=activation,
                    block=block,
                    *args,
                    **kwargs
                ),
                *[
                    ResNetLayer(
                        in_channels * block.expansion,
                        out_channels,
                        n=n,
                        activation=activation,
                        block=block,
                        *args,
                        **kwargs
                    )
                    for (in_channels, out_channels), n in zip(
                        self.in_out_block_sizes, deepths[1:]
                    )
                ],
            ]
        )

    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x


class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """

    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x


class ResNet(nn.Module):
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(
            self.encoder.blocks[-1].blocks[-1].expanded_channels, n_classes
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)


def resnet5(in_channels, n_classes):
    return ResNet(
        in_channels,
        n_classes,
        block=ResNetBasicBlock,
        blocks_sizes=[16, 32],
        deepths=[2, 2],
    )


class ResNetOnLUS(nn.Module):
    def __init__(self):
        super(ResNetOnLUS, self).__init__()
        # load pretrained model for feature extraction
        self.feature_extractor = ResNetEncoder(
            in_channels=3,
            n_classes=1,
            block=ResNetBasicBlock,
            blocks_sizes=[16, 32],
            deepths=[2, 2],
        )
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        num_ftrs = self.feature_extractor.blocks[-1].blocks[-1].expanded_channels

        # embedding layer for position from sites inputs
        embedding_dim = 12
        self.pos_embedding = nn.Embedding(
            num_embeddings=12, embedding_dim=embedding_dim
        )

        # add new fc layers
        self.fc1 = nn.Linear(num_ftrs + embedding_dim, 1024)
        self.fc3 = nn.Linear(1024, 1)

    def forward(self, images, sites, masks):
        # images is a tensor of batch_size x num_sites x 3 x 224 x 224

        feature_vectors = []
        for i in range(images.shape[1]):
            each_image_site = images[:, i, :, :, :]
            each_image_site = each_image_site.view(
                images.shape[0], images.shape[2], images.shape[3], images.shape[4]
            )

            # x is now batch_size x 3 x 224 x 224
            x = self.feature_extractor(each_image_site)
            x = self.avg(x)
            x = torch.squeeze(x)
            feature_vectors.append(x)

        # position embedding of size batch_size x num_sites x embedding_dim
        embedded_sites = self.pos_embedding(sites)

        # stack all feature vectors to a new dimension of size batch_size x num_sites x num_features (512 for ResNet18)
        x = torch.stack(feature_vectors, dim=1)

        # concatenate feature vectors and position embeddings (batch_size x num_sites x [num_features + embedding_dim])
        x = torch.cat([x, embedded_sites], dim=2)

        masks = masks == 1  # convert to boolean
        masks = masks.unsqueeze(-1).expand(
            x.size()
        )  # expand from batch_size x num_sites to batch_size x num_sites x num_features (512 for ResNet18)

        x = x * masks  # apply masks and preserver the original tensor dimensions

        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))

        # average all feature vectors
        x = torch.mean(x, dim=1)  # batch_size x [num_features + embedding_dim]

        # x = self.dropout(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


def get_model():
    return TransferedLearned()


def get_smaller_resnet():
    return ResNetOnLUS()