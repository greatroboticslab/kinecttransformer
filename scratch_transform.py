from multiprocessing import cpu_count
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

import math
from os.path import join
from kinect_learning import * #(joints_collection, load_data, SVM, Random_Forest, AdaBoost, Gaussian_NB, Knn, Neural_Network)
import time

from network import MultitaskTransformerModel


DATA_DIR = 'data'
FILE_NAME = 'bending.csv'
FILE_PATH = join(DATA_DIR, FILE_NAME)

SEED = 1
np.random.seed(SEED)

COLLECTION = joints_collection('bending')
print("Printing scores of small collection...")
print("Collection includes", COLLECTION)
print("Printing scores of small collection with noise data...")
NOISE = False
X = load_data_multiple_dimension(
    FILE_PATH,
    COLLECTION,
    NOISE
)['positions']
Y = load_data_multiple_dimension(
    FILE_PATH,
    COLLECTION,
    NOISE
)['labels']


def create_datasets(x, y, test_size=0.4):
    x = np.asarray(X, dtype=np.float32)
    y = np.asarray(y,dtype=np.int16)

    division = x.shape[0] % 200
    actual_length =  x.shape[0] - division
    x = x[0:actual_length,:]
    y = y[0:actual_length]

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.4)
    x_train, x_valid = [torch.tensor(arr, dtype=torch.float32) for arr in (x_train, x_valid)]
    y_train, y_valid = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_valid)]

    return TensorDataset(x_train, y_train), TensorDataset(x_valid, y_valid)

def create_loaders(train_ds, valid_ds, batch_size=512, jobs=0):
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, batch_size, shuffle=False, num_workers=jobs)
    return train_dl, valid_dl


print('Preparing datasets')
TRN_DS, VAL_DS = create_datasets(X, Y)

BATCH_SIZE = 200
print('Creating data loaders with batch size: {}'.format(BATCH_SIZE))
TRN_DL, VAL_DL = create_loaders(TRN_DS, VAL_DS, BATCH_SIZE, jobs=cpu_count())

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
MODEL = MultitaskTransformerModel(
    task_type='classification',
    device=DEVICE,
    nclasses=2,
    seq_len=2,
    batch=200,
    input_size=3,
    emb_size=4,
    nhead=4,
    nhid=4,
    nhid_tar=4,
    nhid_task=4,
    nlayers=4,
    dropout=0.1
).to(DEVICE)
LR = 0.01

CRITERION = torch.nn.CrossEntropyLoss()
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LR)
N_EPOCHS = 5000

BEST_ACC = 0
PATIENCE, TRIALS = 100000, 0

print('Start model training')

for EPOCH in range(1, N_EPOCHS + 1):
    for _, (X_BATCH, Y_BATCH) in enumerate(TRN_DL):        
        MODEL.train()
        OPTIMIZER.zero_grad()
        try:
            OUT, _ = MODEL(X_BATCH, 'classification')
        except RuntimeError:
            continue
        LOSS = CRITERION(OUT, Y_BATCH)
        LOSS.backward()
        OPTIMIZER.step()

    MODEL.eval()
    CORRECT, TOTAL = 0, 0

    for X_VAL, Y_VAL in VAL_DL:
        X_VAL, Y_VAL = [t for t in (X_VAL, Y_VAL)]
        try:
            OUT, _ = MODEL(X_VAL, 'classification')
        except RuntimeError:
            continue
        PREDS = F.log_softmax(OUT, dim=1).argmax(dim=1)
        TOTAL += Y_VAL.size(0)
        CORRECT += (PREDS == Y_VAL).sum().item()

    print(CORRECT, TOTAL)
    ACC = CORRECT * 1.0 / TOTAL
    print("%.5f" % ACC)

    if EPOCH % 5 == 0:
        VAR = 'Epoch: {:3d}. Loss: {:4.4f}. Acc.: {:2.4f}'
        VAR = VAR.format(EPOCH, LOSS.item(), ACC)
        print(VAR)

    if ACC > BEST_ACC:
        TRIALS = 0
        BEST_ACC = ACC
        torch.save(MODEL.state_dict(), 'best.pth')
        VAR ='Epoch {:2d} best MODEL saved with accuracy: {:2.2f}'
        VAR.format(EPOCH, BEST_ACC)
        print()
    else:
        TRIALS += 1
        if TRIALS >= PATIENCE:
            VAR = 'Early stopping on epoch {}'
            VAR.format(EPOCH)
            print()
            break



