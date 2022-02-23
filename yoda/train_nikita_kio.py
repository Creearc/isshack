
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import pickle

f = open('annotaion.txt', 'r')
s = f.readlines()
f.close()

with open('modules/data.pickle', 'rb') as f:
    arrs = pickle.load(f)

classes = []

points_arr = []
classes_arr = []
CLASSES_NUM = 3

for i in range(len(s)):
  ann = s[i].split(' ')
  frame_number = int(ann[0])
  if frame_number in arrs.keys():
    cl = ann[1][:-1]
    if not (cl in classes):
      classes.append(cl)
    cl = classes.index(cl)
    
    points = arrs[frame_number]
    coords = []
    for elem in points:
      for coord in elem:
        coords.append(coord)
    points_arr.append(coords + [cl])

nms = []

for i in range(len(points_arr[0]) - 1):
  nms.append('X{}'.format(i))

nms.append('Class')              

df = pd.DataFrame(data = points_arr, columns = nms)

print(df)

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = y, random_state=13)



EPOCHS = 1500
BATCH_SIZE = 512
LEARNING_RATE = 0.0000001


class LoadData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)


train_data = LoadData(torch.FloatTensor(np.array(X_train)), 
                       torch.FloatTensor(np.array(y_train)))
  
class TestData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)
    

test_data = LoadData(torch.FloatTensor(np.array(X_test)),torch.FloatTensor(np.array(y_test)))

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE*40, shuffle=False)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE*40)

class NClassifier(nn.Module):
    def __init__(self):
        super(NClassifier, self).__init__()
        self.layer_1 = nn.Linear(66, 132) 
        self.layer_3 = nn.Linear(132, 66)
        self.layer_5 = nn.Linear(66, 10)
        self.layer_out = nn.Linear(10, 3) 
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.005)
        self.batchnorm = nn.BatchNorm1d(132)
        self.sig = nn.Sigmoid()
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm(x)
        x = self.tanh(self.layer_3(x))
        x = self.tanh(self.layer_5(x))
        x = self.dropout(x)
        x = self.sig(self.layer_out(x))
        
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = NClassifier()
model.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

from sklearn.metrics import f1_score

def score(y_p, y_t):
    return f1_score(y_t.cpu().detach().numpy(),
                    np.argmax(y_p.cpu().detach().numpy(), axis=1),average='weighted')


model.train()
for e in range(1, EPOCHS+1):
    
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)

        loss = criterion(y_pred, y_batch.long())
        acc = score(y_pred, y_batch)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        

    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | F1: {epoch_acc/len(train_loader):.3f}')


torch.save(model.state_dict(), 'saved_model.pth')


