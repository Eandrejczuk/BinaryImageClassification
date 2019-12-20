import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
import numpy as np
from Dataset import TrainingDatasetClass
from ImageClassificationConvNet import ImageClassificationConvNet
from torch.utils.data.sampler import SubsetRandomSampler

# Hyperparameters
num_epochs = 21
batch_size = 50
learning_rate = 0.0001
validation_split = 0.2
shuffle_dataset = True
random_seed= 42

MODEL_STORE_PATH = 'C:\\Users\Ewa\Documents\PycharmProjects\\binaryimageclassification\pytorch_models\\'

###############################################  Data Preparation #############################################

#################### Images ####################
array_a_noisy = np.load('data/training/many_noisy_class_a.npy')
array_b_noisy = np.load('data/training/many_noisy_class_b.npy')
crdataset = np.concatenate((array_a_noisy, array_b_noisy))
crdataset = np.expand_dims(crdataset, axis=3)

#################### Labels ####################
labels_a = np.zeros((len(array_a_noisy), 1), dtype = int)
labels_b = np.ones((len(array_b_noisy), 1), dtype = int)
labels = np.concatenate((labels_a,labels_b))

#################### Check all records loaded ####################
assert(len(crdataset) == 20000)
assert(len(labels) == 20000)

trans = transforms.Compose([transforms.ToPILImage(),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.RandomAffine(0, translate=(0.3, 0.3)),
                            transforms.ToTensor()
                            ])

dataset = TrainingDatasetClass(crdataset, labels, trans)

#################### Split data to training and validation ####################

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

#################### Data Loader ####################
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)


###############################################  Training #############################################
cnn_model = ImageClassificationConvNet()

#################### Loss and optimizer ####################
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate)

#################### Train the model ####################
total_step = len(train_loader)
loss_list = []
accuracy_list = []
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # forward pass
        outputs = cnn_model(images.float())
        labels = labels.squeeze(1)
        loss = criterion(outputs, labels.long())
        loss_list.append(loss.item())

        # Backprop and Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        #print("predicted", predicted, "labels", labels)
        correct = (predicted == labels).sum().item()
        accuracy_list.append(correct / total)

        if (i + 1) % (int((dataset_size - split) / (batch_size*2))) == 0:
            #print("correct: ", correct, " / total: ", total)
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))

###############################################  Evaluating the model #############################################
cnn_model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = cnn_model(images.float())
        _, predicted = torch.max(outputs.data, 1)
        labels = labels.squeeze(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the {} test images: {} %'.format(split, (correct / total) * 100))

# Save the model and plot
torch.save(cnn_model.state_dict(), MODEL_STORE_PATH + 'conv_net_model.ckpt')

p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch Image Classification ConvNet results')
p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
p.line(np.arange(len(loss_list)), loss_list)
p.line(np.arange(len(loss_list)), np.array(accuracy_list) * 100, y_range_name='Accuracy', color='red')
show(p)