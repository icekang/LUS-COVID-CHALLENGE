import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
        # self.dropout = nn.Dropout(0.10)
        # self.fc2 = nn.Linear(256, 128)

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


def get_model():
    return Net()