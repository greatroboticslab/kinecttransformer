# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:09:45 2021

@author: Ranak Roy Chowdhury
"""
import utils, argparse, warnings, sys, shutil, torch, os, numpy as np, math
warnings.filterwarnings("ignore")
from os.path import join
from torch.utils.data import TensorDataset, DataLoader
from kinect_learning import * #(joints_collection, load_data, SVM, Random_Forest, AdaBoost, Gaussian_NB, Knn, Neural_Network)

DATA_DIR = 'data'
FILE_NAME = 'bending.csv'
FILE_PATH = join(DATA_DIR, FILE_NAME)


def create_datasets(x, y, test_size=0.4):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)

    division = x.shape[0] % 200
    actual_length =  x.shape[0] - division
    x = x[0:actual_length,:]
    y = y[0:actual_length]

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.4)


    return (x_train, y_train), (x_valid, y_valid)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AF')
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--emb_size', type=int, default=64)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--task_rate', type=float, default=0.5)
parser.add_argument('--masking_ratio', type=float, default=0.15)
parser.add_argument('--lamb', type=float, default=0.8)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--ratio_highest_attention', type=float, default=0.5)
parser.add_argument('--avg', type=str, default='macro')
parser.add_argument('--dropout', type=float, default=0.01)
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--nhid_task', type=int, default=128)
parser.add_argument('--nhid_tar', type=int, default=128)
parser.add_argument('--task_type', type=str, default='classification', help='[classification, regression]')
args = parser.parse_args()



def main():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    prop = utils.get_prop(args)
    prop['task_type'] = 'classification'
    prop['device'] = DEVICE
    prop['nclasses'] = 2
    prop['seq_len'] = 2
    prop['batch'] = 200
    prop['input_size'] = 3
    prop['emb_size'] = 4
    prop['nhead'] = 4
    prop['nhid'] = 4
    prop['nhid_tar'] = 4
    prop['nhid_task'] = 4
    prop['nlayers'] = 4
    prop['dropout'] = 0.1

    path = './data/' + prop['dataset'] + '/'
    print('Data loading start...')
    X_train, y_train, X_test, y_test = utils.data_loader(args.dataset, path, prop['task_type'])
    print('Data loading complete...')
    print([item.shape for item in [X_train, y_train, X_test, y_test]])
    print([item.dtype for item in [X_train, y_train, X_test, y_test]])
    print('A1: {}'.format(type(X_train)))



    print('Data loading start...')
    COLLECTION = joints_collection('bending')
    print("Printing scores of small collection...")
    print("Collection includes", COLLECTION)
    print("Printing scores of small collection with noise data...")
    NOISE = False
    DATA = load_data_multiple_dimension(
        FILE_PATH,
        COLLECTION,
        NOISE
    )
    X, Y = DATA['positions'], DATA['labels']
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.int64)

    divsion = X.shape[0] % 10
    actual_length =  X.shape[0] - divsion
    X = X[0:actual_length,:]
    Y = Y[0:actual_length]
    print('HERE: {}'.format((X.shape, Y.shape)))
    (X_train, y_train), (X_test, y_test) = create_datasets(X, Y)
    print('Data loading complete...')
    print([item.shape for item in [X_train, y_train, X_test, y_test]])
    print([item.dtype for item in [X_train, y_train, X_test, y_test]])
    print('A2: {}'.format(type(X_train)))
    

    print('Data preprocessing start...')
    X_train_task, y_train_task, X_test, y_test = utils.preprocess(prop, X_train, y_train, X_test, y_test)
    print(X_train_task.shape, y_train_task.shape, X_test.shape, y_test.shape)
    print('Data preprocessing complete...')

    prop['nclasses'] = torch.max(y_train_task).item() + 1 if prop['task_type'] == 'classification' else None
    prop['dataset'], prop['seq_len'], prop['input_size'] = prop['dataset'], X_train_task.shape[1], X_train_task.shape[2]
    prop['device'] = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    print('Initializing model...')
    model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer = utils.initialize_training(prop)
    print('Model intialized...')

    print('Training start...')
    utils.training(model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer, X_train_task, y_train_task, X_test, y_test, prop)
    print('Training complete...')



if __name__ == "__main__":
    main()
