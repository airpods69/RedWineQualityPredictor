import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from model import Model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.metrics import accuracy_score

wine = np.loadtxt('./winequality-red.csv',
                  delimiter = ",",
                  skiprows = 1)

test_per = 0.2 # test percentage = 20%

n_features = 11 # No of Features

X = wine[:, :n_features]
X = scale(X)

label = wine[:, n_features:]

oneHot = OneHotEncoder()

label = oneHot.fit_transform(label).toarray()

X_train , X_test , Y_train , Y_test = train_test_split(X , label, test_size = test_per)


model = Model(n_features)

criterion = nn.BCELoss()

optimizer = optim.SGD(model.parameters(),
                      lr = 0.5)

X = Variable(torch.from_numpy(X_train).float())
Y = Variable(torch.from_numpy(Y_train).float())

n = 0
losses = []
act_class = []
pred_class = []
train_acc = []

for epoch in range(10000):

    out = model(X)
    loss = criterion(out, Y)
    losses.append(loss)

    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    train_acc.append(accuracy_score(oneHot.inverse_transform(Y), oneHot.inverse_transform(model(X).data.numpy())))

    if epoch % 50 == 0:
        print("Trained {} epochs and loss: {}".format(epoch, loss.data))


with torch.no_grad():
    losses = np.array(losses, dtype=float)


train_out = oneHot.inverse_transform(model(X).data.numpy())
print('Training accuracy', accuracy_score(oneHot.inverse_transform(Y), train_out))

test_out = oneHot.inverse_transform(model(torch.from_numpy(X_test).float()).data.numpy())
print('prediction accuracy', accuracy_score(oneHot.inverse_transform(Y_test), test_out))

