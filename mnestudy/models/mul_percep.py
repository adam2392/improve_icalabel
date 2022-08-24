import os
import pandas as pd
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
raw_data = training_data_topos('C:/Users/asaini/Desktop/FCBG/iclabel-python/BIDS_DATA/MARA_test')
x = torch.stack([i[0] for i in raw_data])   # Tensor shape (subjects, 30, 64, 64)
y = torch.stack([i[1] for i in raw_data])   # Tensor shape (subjects, 30)


x_data = x.view(-1, *x.size()[2:])  # same as x_t = x.view(27*30, 64,64) # Tensor shape (subjects*30, 64, 64)
y_data = y.view(-1)                 # # Tensor shape (subjects*30,)

data = torch.utils.data.TensorDataset(x_data, y_data)

BATCH_SIZE = 30    # size of a batch in training
data_loader = torch.utils.data.DataLoader(data, batch_size = BATCH_SIZE)

class Single_layer_net(torch.nn.Module):

    def __init__(self, num_features):
        super(Single_layer_net, self).__init__()

        self.linear = torch.nn.Linear(num_features,256)
        self.linear.weight.detach().normal_(0.0, 0.1)
        self.linear.bias.detach().zero_()

        self.linear_2 = torch.nn.Linear(256,1)
        self.linear_2.weight.detach().normal_(0.0, 0.1)
        self.linear_2.bias.detach().zero_()

    def forward(self, x):
        out = self.linear(x)
        out = self.linear_2(out)
        logits = torch.sigmoid(out)
        return logits

RANDOM_SEED = 1
torch.manual_seed(RANDOM_SEED)
model = Single_layer_net(num_features=64*64).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss = BCELoss()

def compute_loss(model, data_loader):
    curr_loss = 0.
    # with torch.no_grad():
    for cnt, (features, targets) in enumerate(data_loader):
        features = features.view(-1, 64*64).to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(features)
        loss = BCELoss()
        loss = loss(logits, targets.unsqueeze(1).float())
        curr_loss += loss
    return float(curr_loss)/cnt

NUM_EPOCHS = 100


start_time = time.time()
minibatch_cost = []
epoch_cost = []
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(data_loader):

        features = features.view(-1, 64*64).to(DEVICE)
        targets = targets.unsqueeze(1).float().to(DEVICE)

        logits = model(features)
        # _, pred = torch.max(logits, 1)
        cost = loss(logits, targets)
        optimizer.zero_grad()

        cost.backward()

        optimizer.step()


        minibatch_cost.append(cost.item())
        print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                   %(epoch+1, NUM_EPOCHS, batch_idx,
                     len(data_loader), cost.item()))

    cost = compute_loss(model, data_loader)
    epoch_cost.append(cost)
    print('Epoch: %03d/%03d Train Cost: %.4f' % (
            epoch+1, NUM_EPOCHS, cost))
    print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
    print('\n')

print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

plt.plot(range(len(minibatch_cost)), minibatch_cost)
plt.ylabel('Cross Entropy')
plt.xlabel('Minibatch')
plt.show()

plt.plot(range(len(epoch_cost)), epoch_cost)
plt.ylabel('Cross Entropy')
plt.xlabel('Epoch')
plt.show()


def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    for features, targets in data_loader:
        features = features.view(-1, 64*64).to(DEVICE)
        targets = targets.to(DEVICE)
        logits = model(features)
        predicted_labels = torch.argmax(logits, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100

print('Training Accuracy: %.2f' % compute_accuracy(model, data_loader))
