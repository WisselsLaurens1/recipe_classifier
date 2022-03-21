from pathlib import Path
import pickle
import sys
from nltk.util import pr
sys.path.append('../')
from sklearn.metrics import accuracy_score
from scripts.tools import _train_test_split, cap_dataset, remove_invalid
from scripts.tools import data_loader
import torch
import torch.nn as nn
import pandas as pd
import json


TRAIN = False
INVALID = True
base_dir = Path(__file__).resolve().parents[1]
NN_PATH = f'{base_dir}/models/NeuralNetwork.pth'
if INVALID:
    NN_PATH = f'{base_dir}/models/NeuralNetwork_with_invalid.pth'

if INVALID:
    print('This includes invalid')
else:
    print('This does not include invalid')

print('Loading data')
file = open('../data/train.sav', 'rb')
# dump information to that file
data = pickle.load(file)
# close the file
file.close()
print('Loaded data')

# Y = df['cuisine'].values
Y = data["cuisines"]


# df = df.drop(['cuisine','index'],axis=1)

# labels = list(df.columns)
# X = df[labels].values

X = data["vectors"]

cuisines = list(set(Y))

if INVALID:
    cuisines.append("invalid")

cuisines.sort()
Y = [cuisines.index(y) for y in Y]

X,Y = remove_invalid(X,Y)

# #balance dataset
# X, Y = cap_dataset(X,Y,limit=450)

X_train, X_test, Y_train, Y_test = _train_test_split(X, Y,test_size=0.03)

train_loader = data_loader(X=X_train,Y=Y_train,batch_size=10)

X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train)


class NeurelNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeurelNetwork, self).__init__()

        self.l1 = nn.Linear(input_size, 150)
        self.l2 = nn.Linear(150,125)
        self.l3 = nn.Linear(125,125)
        self.l4 = nn.Linear(125, output_size)

    def forward(self, X):
        out = torch.rrelu(self.l1(X))
        out = torch.sigmoid(self.l2(out))
        out = torch.rrelu(self.l3(out))
        out = torch.rrelu(self.l4(out))
        # softmax for training is included in cost function

        return out


n_features = len(X[0])
n_outputs = len(cuisines)
# n_outputs = len(ground_thruts)

learning_rate = 0.1

model = NeurelNetwork(n_features, n_outputs)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print('#'*20)
print('\n')
if TRAIN:

    e = 0
    for epoch in range(50):
        for i, (x, y) in enumerate(train_loader):

            Y_pred = model(x)
            l = loss(Y_pred, y)

            if i % 30 == 0:
                sys.stdout.write(f'epoch:{e}, loss:{l}\r')
                sys.stdout.flush()

            l.backward()
            optimizer.step()
            optimizer.zero_grad()

        e += 1

    print(f'\nSaving model to {NN_PATH}')
    torch.save(model.state_dict(), NN_PATH)



else:
    print('Loading model')
    model = NeurelNetwork(n_features, n_outputs)
    model.load_state_dict(torch.load(NN_PATH))
    model.eval()
    print('Loaded model')



Y_pred = model(X_train)

_, predicted = torch.max(Y_pred.data, 1)

score = accuracy_score(predicted, Y_train)
print(f'Train accuracy: {score}')

with torch.no_grad():

    df = pd.read_csv('../data/cleaned_validation.csv')


    print('Loading data')
    file = open('../data/validation.sav', 'rb')
    # dump information to that file
    data = pickle.load(file)
    # close the file
    file.close()
    print('Loaded data')

    Y = data["cuisines"]


    # df = df.drop(['cuisine'],axis=1)
    # labels = list(df.columns)
    # X = df[labels].values

    X = data["vectors"]

    if not INVALID:
        X,Y = remove_invalid(X,Y)

    Y = [cuisines.index(y) for y in Y]


    X = torch.tensor(X,dtype=torch.float32)
    Y = torch.tensor(Y)

    Y_pred = model(X)
    # Y_pred = torch.softmax(Y_pred)
 
    # print(Y_pred)

    sm = torch.nn.Softmax()
    probabilities = sm(Y_pred) 
    print(probabilities[0])



    for prob,y in zip(probabilities,Y):
        if y != cuisines.index("invalid"):
            print(prob)
            break



    # l = loss(Y_pred, Y)

    # _, predicted = torch.max(Y_pred.data, 1)

    # score = accuracy_score(predicted, Y)
    # print(f'Test accuracy: {score}')
