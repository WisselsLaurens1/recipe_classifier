import csv
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import math

def class_ratio(Y):
    ratios = {}
    result = []
    tot = 0
    for y in Y:
        ratios.setdefault(y, 0)
        ratios[y] += 1
        tot += 1
    for key in ratios:
        ratios[key] = ratios[key]/tot
        result.append((key, tot*ratios[key], ratios[key]))

    result.sort(key=lambda x: x[1], reverse=True)
    return result


def get_class_ratio(Y,_print=True):
    ratios = class_ratio(Y)
    if _print:
        for r in ratios:
            print(f'* {r[0]} {int(r[1])}')
    else:
        return ratios


def create_voc(x: list) -> list:
    voc = []
    for sample in x:
        for word in sample:
            voc.append(word)

    voc = list(set(voc))
    voc.sort(reverse=True)
    return voc

def remove_invalid(X,Y):
    tmp_X = []
    tmp_Y = []
    for x,y in zip(X,Y):
        if y != 'invalid':
            tmp_X.append(x)
            tmp_Y.append(y)
    return tmp_X,tmp_Y


def vectorize(x: list, voc: list) -> list:

    vectors = []
    placeholder = [0 for i in voc]
    for sample in x:
        vector = placeholder.copy()
        for word in sample:
            if word in voc:
                vector[voc.index(word)] = 1
        vectors.append(vector)

    return vectors


def cap_dataset(X, Y, limit):
    class_limits = {y: 0 for y in list(set(Y))}

    tmp_X = []
    tmp_Y = []

    for x, y in zip(X, Y):
        if class_limits[y] < limit:
            tmp_X.append(x)
            tmp_Y.append(y)
            class_limits[y] += 1

    return tmp_X, tmp_Y


def write(X: list, Y: list, headers: list, path: str):
    '''
    X: vectorized features
    Y: labels
    headers: list of headers
    '''

    data = []
    for x, y in zip(X, Y):
        data.append(x+[y])

    with open(path, 'w') as f:
        writer = csv.writer(f)

        headers.append('cuisine')
        writer.writerow(headers)
        writer.writerows(data)


def show_wrongly_predicted(X, Y, model):

    predictions = model.predict(X)

    wrong_pred = {}
    for pred, y in zip(predictions, Y):
        if pred != y:
            wrong_pred.setdefault(y, 0)
            wrong_pred[y] += 1

    train_ratios = class_ratio(Y)
    # for r in train_ratios:
    #     print(f'*{r[0]} {r[1]}')

    cr = {r[0]: int(r[1]) for r in train_ratios}

    print(' ')
    # n times wrongly classified
    for y in wrong_pred:
        wa = wrong_pred[y]
        print(
            f'* wrongly predicted {y}: {wa}/{cr[y]}:  {int(round(wa/cr[y],3)*100)}%')



class _Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # self.X = torch.from_numpy(X, dtype=torch.float32)
        # self.Y = torch.from_numpy(Y, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32).long()
        # self.Y = torch.tensor(Y, dtype=torch.float32)
        # self.Y = self.Y.view(self.Y.shape[0],1)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.Y.shape[0]


'''
X: python list
Y: python list
'''


# def train_test_loader(X: list, Y: list, train_batch_size: int, test_batch_size: int):
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y,random_state=1,shuffle=False)

#     train_set = _Dataset(X_train, Y_train)
#     test_set = _Dataset(X_test, Y_test)

#     train_loader = DataLoader(train_set, batch_size=train_batch_size,shuffle=False)
#     test_loader = DataLoader(test_set, batch_size=test_batch_size,shuffle=False)

#     datasets = [{'X':X_train,'Y':Y_train},{'X':X_test,'Y':Y_test}]

#     return train_loader, test_loader, datasets


def data_loader(X:list,Y:list,batch_size):
    data_set = _Dataset(X,Y)    

    _data_loader = DataLoader(dataset=data_set,batch_size=batch_size)

    return _data_loader


def _train_test_split(X,Y,test_size):
    i  = math.ceil(len(X)*(1-test_size))
    X_train = X[0:i] 
    Y_train = Y[0:i]
    X_test = X[i+1:]
    Y_test = Y[i+1:]
    return X_train,X_test,Y_train,Y_test   
