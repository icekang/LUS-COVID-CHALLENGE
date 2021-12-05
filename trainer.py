
import matplotlib.pyplot as plt
import ml_collections 
import deepchest
import os
# ## Model Configuration

config = ml_collections.ConfigDict()

config.batch_size = 32
config.num_steps = 300

# See preprocessing.py, if you replace with ";" no preprocessing is done
config.preprocessing_train_eval = "independent_dropout(.2);"

config.use_validation_split = False

# If validation split is false, then train will have 4/5 of data and test 1/5
# If validation split is true, then train will have 3/5 of data, test 1/5 and val 1/5
config.num_folds = 5 

# gpu workers
config.num_workers = 0

# dataset 
config.images_directory = "dataset/images/"
config.labels_file = "dataset/labels/diagnostic.csv"

# Fold seed
config.random_state = 0

# Where the indices are saved
config.save_dir = "model_saved/"
config.export_folds_indices_file = "indices.csv"

# Don't modify these (should not have been in the config)
config.test_fold_index = 0
config.delta_from_test_index_to_validation_index = 1
# # Data Preprocessing

train_loader, test_loader, validation_loader = deepchest.dataset.get_data_loaders(
         config=config
    )

# How to evaluate model (you can also use your own methods)
# ```python
# label_names = utils.get_label_names(config.labels_file)
# 
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:0") if use_cuda else torch.device("cpu") 
# 
# scores, labels = utils.model_evaluation(**model**, train_loader, device)
# train_metrics = utils.compute_metrics(labels, scores, label_names)
# 
# train_metrics
# ```
# # Apply simple model

import copy
import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

if torch.cuda.is_available(): print("CUDA is available")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(config.random_state)

# device = torch.device("cuda:0" if False else "cpu")

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
        self.pos_embedding = nn.Embedding(num_embeddings=12, embedding_dim=embedding_dim)

        # add new fc layers
        # self.fc1 = nn.Linear(num_ftrs + embedding_dim, 256)
        # self.dropout = nn.Dropout(0.10)
        # self.fc2 = nn.Linear(256, 128)

        self.fc3 = nn.Linear(num_ftrs + embedding_dim, 1)

    def forward(self, images, sites, masks):
        # images is a tensor of batch_size x num_sites x 3 x 224 x 224

        feature_vectors = []
        for i in range(images.shape[1]):
            each_image_site = images[:, i, :, :, :]
            each_image_site = each_image_site.view(images.shape[0], images.shape[2], images.shape[3], images.shape[4])

            # x is now batch_size x 3 x 224 x 224
            x = self.feature_extractor(each_image_site)
            feature_vectors.append(x)

        # position embedding of size batch_size x num_sites x embedding_dim
        embedded_sites = self.pos_embedding(sites)

        # stack all feature vectors to a new dimension of size batch_size x num_sites x num_features (512 for ResNet18)
        x = torch.stack(feature_vectors, dim=1)

        # concatenate feature vectors and position embeddings (batch_size x num_sites x [num_features + embedding_dim])
        x = torch.cat([x, embedded_sites], dim=2)


        masks = masks == 1 # convert to boolean
        masks = masks.unsqueeze(-1).expand(x.size()) # expand from batch_size x num_sites to batch_size x num_sites x num_features (512 for ResNet18)

        x = x * masks # apply masks and preserver the original tensor dimensions

        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))

        # average all feature vectors
        x = torch.mean(x, dim=1) # batch_size x [num_features + embedding_dim]

        # x = self.dropout(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x

def train_model(model, criterion, optimizer, scheduler, config, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            
            data_loader = {'train': train_loader, 'test': test_loader, 'val': test_loader}[phase]
            max_iteration = len(data_loader)
            # Iterate over data.
            batch_i = 1
            image_count = 0
            for batch in data_loader:
                images, sites, masks, labels = batch['images'], batch['sites'], batch['mask'], batch['label']
                image_count += images.shape[0]
                images = images.to(device)
                sites = sites.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images, sites, masks)
                    outputs = torch.squeeze(outputs)
                    preds = torch.zeros_like(outputs)
                    preds[outputs > 0.5] = 1
                    preds[outputs <= 0.5] = 0
                    loss = criterion(outputs, labels.float())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # print('model.fc3.weight.data', model.fc3.weight.data)

                # statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print(f'iteration: {batch_i}, running_loss: {running_loss}, correct: {running_corrects}' + ' ' * 10 , end='\r')
                batch_i += 1
                
                if image_count >= max_iteration:
                # if batch_i >= config.num_steps:
                    break
                # print('preds', preds, '\nlabels.data', labels.data)
            if phase == 'train':
                scheduler.step()
            print('running_corrects', running_corrects)
            epoch_loss = running_loss / image_count
            epoch_acc = running_corrects / image_count

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('Last prediction', preds, '\nLabels.data', labels.data)

            # deep copy the model
            if (phase == 'val') and (epoch_acc > best_acc):
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss,
                    'acc': epoch_acc,
                }, os.path.join(config.save_dir, f'model_resnet50_sigmoid_epoch{epoch}'))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# ## Train model

model = Net().to(device)
print(model)

criterion = nn.BCELoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
params = list(model.fc3.parameters()) + list(model.pos_embedding.parameters())
# params = list(model.fc1.parameters()) + list(model.fc2.parameters()) + list(model.fc3.parameters()) + list(model.pos_embedding.parameters())
optimizer_conv = optim.Adam(params, lr=1e-4)

# Decay LR by a factor of 0.1 every 4 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=5, gamma=0.1)

model_conv = train_model(model, criterion, optimizer_conv, exp_lr_scheduler, config, num_epochs=50)

# saving the model
torch.save(model_conv.state_dict(), f'{config.save_dir}simple_model')


# ## Evaluation

model = Net()
model.load_state_dict(torch.load(f'{config.save_dir}simple_model'))
model.eval()
model.to(device)

label_names = deepchest.utils.get_label_names(config.labels_file)

use_cuda = torch.cuda.is_available()
# use_cuda = False

device = torch.device("cuda:0") if use_cuda else torch.device("cpu")

scores, labels = deepchest.utils.model_evaluation(model, test_loader, device)
train_metrics = deepchest.utils.compute_metrics(labels, scores, label_names)

train_metrics

