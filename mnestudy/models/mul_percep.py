import os
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from mne.preprocessing import read_ica
from mne_icalabel.features.topomap import get_topomaps
from torch import nn
from sklearn.preprocessing import LabelEncoder
from torch.nn.functional import binary_cross_entropy
from torch.nn import BCELoss
import time
from torch.nn.functional import relu

######################################
# DATA LOADING
######################################

# Uploading ICA_files and Labels from the folder in BIDS format
def training_data_topos(directory):
    train_data = []
    for root, subdirectories, files in os.walk(directory):
        for file in files:
            os.chdir(root)
            l = list()
            if file.endswith('ica.fif') or file.endswith('ica_markers.fif'):

                ica_file = read_ica(file)
                topomap_arrays = get_topomaps(ica_file)
            if file.endswith('markers.tsv'):
                labels = pd.read_csv(file, sep= '\t').iloc[:, 3]
                label_encoder = LabelEncoder()
                labels = label_encoder.fit_transform(labels) # Encoding the labels
                labels = torch.LongTensor(labels)
                if topomap_arrays.any():
                    l.append(torch.Tensor(topomap_arrays))
                    l.append(labels)
                    train_data.append(l)
                else:
                    continue
    return train_data # returns a list of lists containing Topomap feature Tensor and corrresponding labels Tensor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Processing lists of lists into a tensor by stacking the lists vertically
raw_data = training_data_topos('C:/Users/asaini/Desktop/FCBG/iclabel-python/BIDS_DATA/MARA_test_full')
x = torch.stack([i[0] for i in raw_data])   # Tensor shape (subjects, 30, 64, 64)
y = torch.stack([i[1] for i in raw_data])   # Tensor shape (subjects, 30)
x_data = x.view(-1, *x.size()[2:])  # same as x_t = x.view(27*30, 64,64) # Tensor shape (subjects*30, 64, 64)
y_data = y.view(-1)                 # # Tensor shape (subjects*30,)

# Shuffle the dataset
torch.manual_seed(123)
shuffle_idx = torch.randperm(y_data.size(0))
x_data, y_data = x_data[shuffle_idx], y_data[shuffle_idx]

# Train test split

train_split_80 = int(shuffle_idx.size(0) * 0.8)
x_train, x_test = x_data[shuffle_idx[:train_split_80]], x_data[shuffle_idx[train_split_80:]]
y_train, y_test = y_data[shuffle_idx[:train_split_80]], y_data[shuffle_idx[train_split_80:]]
train_data = torch.utils.data.TensorDataset(x_train, y_train)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
BATCH_SIZE = 30    # size of a batch in training = 30
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size = BATCH_SIZE)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE)

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

# fig,ax = plt.subplots(1,7)
# for i in range(7):
#     ax[i].imshow(x_train[i])
#     ax[i].axis('off')
# print([y_train[i] for i in range(7)])

#####################################
# MODEL
#####################################
class multi_layer_net(torch.nn.Module):

    def __init__(self, num_features):
        super(multi_layer_net, self).__init__()

        self.linear = torch.nn.Linear(num_features,256)
        self.linear.weight.detach().normal_(0.0, 0.1)
        self.linear.bias.detach().zero_()

        self.linear_2 = torch.nn.Linear(256,128)
        self.linear_2.weight.detach().normal_(0.0, 0.1)
        self.linear_2.bias.detach().zero_()


        self.linear_3 = torch.nn.Linear(128,64)
        self.linear_3.weight.detach().normal_(0.0, 0.1)
        self.linear_3.bias.detach().zero_()

        self.linear_4 = torch.nn.Linear(64,1)
        self.linear_4.weight.detach().normal_(0.0, 0.1)
        self.linear_4.bias.detach().zero_()
    def forward(self, x):
        out = self.linear(x)
        out = self.linear_2(out)
        out = relu(out)
        out = self.linear_3(out)
        out = relu(out)
        out = self.linear_4(out)
        logits = torch.sigmoid(out)
        return logits

RANDOM_SEED = 21
torch.manual_seed(RANDOM_SEED)
model = multi_layer_net(num_features=64*64).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss = BCELoss()

def compute_loss(model, data_loader):
    curr_loss = 0.
    # with torch.no_grad():
    for count, (features, targets) in enumerate(data_loader):
        features = features.view(-1, 64*64).to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(features)
        loss = BCELoss()
        loss = loss(logits, targets.unsqueeze(1).float())
        curr_loss += loss
    return float(curr_loss)/count

def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features.view(-1, 64*64).to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(features)
        predicted_labels = torch.where((logits >0.7), 1, 0)
        total_IC = targets.size(0)
        count_correct = (predicted_labels.squeeze(1) == targets).sum()
        # predictions_data = pd.DataFrame({'Targets': targets.detach().numpy(), 'Logits': logits.detach().numpy().squeeze(1), 'Pred_labels': predicted_labels.detach().numpy().squeeze(1)})
        print((count_correct.float()/total_IC) * 100)
    return (count_correct.float()/total_IC) * 100

####################################
# TRAINING
####################################

NUM_EPOCHS = 100
start_time = time.time()
minibatch_cost = []
epoch_cost = []
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_data_loader):
        features = features.view(-1, 64*64).to(DEVICE)
        targets = targets.unsqueeze(1).float().to(DEVICE)
        logits = model(features)
        # _, pred = torch.max(logits, 1)
        cost = loss(logits, targets)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        minibatch_cost.append(cost.item())
        print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f,'
                   %(epoch+1, NUM_EPOCHS, batch_idx+1,
                     len(train_data_loader), cost.item()), )

    cost = compute_loss(model, train_data_loader)
    epoch_cost.append(cost)
    print('Epoch: %03d/%03d Train Cost: %.4f' % (
            epoch+1, NUM_EPOCHS, cost))
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    print('\n')

plt.plot(range(len(minibatch_cost)), minibatch_cost)
plt.ylabel('Loss')
plt.xlabel('Minibatch')
plt.show()
plt.plot(range(len(epoch_cost)), epoch_cost)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

print('Training Accuracy: %.2f' % compute_accuracy(model,train_data_loader))

def validate(net, dataloader, loss):
    net.eval()
    count,acc,test_cost = 0,0,0
    with torch.no_grad():
        for features,labels in dataloader:
            features = features.view(-1, 64*64).to(DEVICE)
            labels = labels.unsqueeze(1).float().to(DEVICE)
            out = net(features)
            test_cost += loss(out,labels)
            pred = torch.where((out >0.7), 1, 0)
            acc += (pred==labels).sum()
            count += len(labels)
    return test_cost.item()/count, (acc.item()/count)*100

cost_test , accuracy_test = validate(model,test_data_loader, loss=loss)
print('Test Loss: {}'.format(cost_test))
print('Test_accuracy: {}'.format(accuracy_test))
