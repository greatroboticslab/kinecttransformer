'''Contains parameters and functions for all general usages.'''

import random
import json
import numpy as np
from pprint import pprint
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#from sklearn.gaussian_process import GaussianProcess
from sklearn.model_selection import cross_validate
from pybrain.datasets 			 import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer


# "left-right" : ['KneeRight', 'KneeLeft', 'AnkleRight', 'AnkleLeft', 'FootRight', 'FootLeft'],

def joints_collection(posture):
    switcher = {
        "left-right" : ['KneeRight', 'KneeLeft', 'AnkleRight', 'AnkleLeft', 'FootRight', 'FootLeft'],
        "turning" : ['HandLeft', 'HandRight', 'WristLeft', 'WristRight', 'ElbowLeft', 'ElbowRight', 'ShoulderLeft',
                    'ShoulderRight', 'ShoulderCenter', 'HipLeft', 'HipRight', 'HipCenter', 'KneeLeft', 'KneeRight'],
        "bending" : ['Head', 'ShoulderLeft', 'ShoulderRight', 'ShoulderCenter', 'ElbowLeft', 'ElbowRight', 'WristLeft',
                   'WristRight', 'HandLeft', 'HandRight', 'Spine', 'HipLeft', 'HipRight', 'HipCenter'],
		"bending1": ['Head', 'ShoulderLeft', 'ShoulderRight', 'ShoulderCenter', 'ElbowLeft', 'ElbowRight', 'WristLeft',
					'WristRight', 'HandLeft', 'HandRight', 'Spine', 'HipLeft', 'HipRight', 'HipCenter'],
        "up-down" : ['HandLeft', 'HandRight', 'WristLeft', 'WristRight', 'ElbowLeft', 'ElbowRight', 'ShoulderLeft',
                   'ShoulderRight', 'ShoulderCenter'],
        "sit-stand" : ['HipCenter', 'HipLeft', 'HipRight', 'KneeRight', 'KneeLeft', 'WristRight', 'WristLeft',
                     'HandRight', 'HandLeft', 'ElbowRight', 'ElbowLeft'],
        "all" : ['HipCenter', 'Spine', 'ShoulderCenter', 'Head', 'ShoulderLeft', 'ElbowLeft', 'WristLeft',
                      'HandLeft', 'ShoulderRight', 'ElbowRight', 'WristRight', 'HandRight', 'HipLeft', 'KneeLeft',
                      'AnkleLeft', 'FootLeft', 'HipRight', 'KneeRight', 'AnkleRight', 'FootRight']
    }
    return switcher.get(posture)


def load_data(file_name, collection, noise):
	with open(file_name) as json_file:
		data = json.loads(json_file.read())
	X = []
	y = []
	count = 0
	length = len(data)
	for datum in data:
		yi = datum['label']
        ## Add 30 noise data or not

		if yi == 2:
			if noise == True:
				count += 1
				if count > 100:
					continue
				else:
					yi = random.randint(0,1)
			else:
				continue

		y.append(yi)
		Xi = []
		features = datum['jointPositions']['jointPositionDict']
		for joint in collection:
			xj = list(features[joint].values())
			#if noise == True:
			#	xj[0] += random.randint(0, 1)
			#	xj[1] += random.randint(0, 1)
			#	xj[2] += random.randint(0, 1)
			Xi = Xi + xj

		X = X + [Xi]
	return {'positions' : X, 'labels' : y}

def load_data_multiple_dimension(file_name, collection, noise):
	with open(file_name) as json_file:
		data = json.loads(json_file.read())
	X = []
	y = []
	count = 0
	length = len(data)
	for datum in data:
		yi = datum['label']
        ## Add 30 noise data or not
		if yi == 2:
			if noise == True:
				count += 1
				if count > 2000:
					continue
				else:
					yi = random.randint(0,1)
			else:
				continue

		y.append(yi)
		Xi = []
		features = datum['jointPositions']['jointPositionDict']
		i = 0
		for joint in collection:
			xj = list(features[joint].values())
			#if noise == True:
			#	xj[0] += random.randint(0, 1)
			#	xj[1] += random.randint(0, 1)
			#	xj[2] += random.randint(0, 1)
			Xi.append(xj)
		X.append(Xi)
	return {'positions' : X, 'labels' : y}

def SVM(X, y, tst_size, ker):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = tst_size, random_state = 0)
	svc = svm.SVC(kernel = ker)
	score = 0
	for i in range(100):
		svc.fit(X_train, y_train)
		score += svc.score(X_test, y_test)
	score = score/100
	return score

def Random_Forest(X, y, tst_size, n_est):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = tst_size, random_state = 0)
	rfc = RandomForestClassifier(n_estimators = n_est)
	score = 0
	for i in range(100):
		rfc.fit(X_train, y_train)
		score += rfc.score(X_test, y_test)
	score = score/100
	return score

def AdaBoost(X, y, tst_size, n_est):
	X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = tst_size, random_state = 0)
	clf = AdaBoostClassifier(n_estimators = n_est)
	score = 0
	for i in range(100):
		clf.fit(X_train, y_train)
		score += clf.score(X_test, y_test)
	score = score/100
	return score

def Gaussian_NB(X, y, tst_size):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = tst_size, random_state = 0)
	clf = GaussianNB()
	score = 0
	for i in range(100):
		clf.fit(X_train, y_train)
		score += clf.score(X_test, y_test)
	score = score/100
	return score

def Knn(X, y, tst_size, num_neighbors):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = tst_size, random_state = 0)
	neigh = KNeighborsClassifier(n_neighbors=num_neighbors)
	score = 0
	for i in range(100):
		neigh.fit(X_train, y_train)
		score += neigh.score(X_test, y_test)
	score = score/100
	return score


def Neural_Network(X, y, tst_size, col_size, n_epochs):
	## Load the dataset into the neural network.
	ds = ClassificationDataSet(3*col_size, 1)
	for i in range(len(X)):
		ds.addSample(X[i], y[i])
	## Split the data into training and testing.
	tstdata_tmp, trndata_tmp = ds.splitWithProportion(tst_size)
	tstdata = ClassificationDataSet(3*col_size, 1, nb_classes = 2)
	for n in range(0, tstdata_tmp.getLength()):
		tstdata.addSample( tstdata_tmp.getSample(n)[0], tstdata_tmp.getSample(n)[1] )
	trndata = ClassificationDataSet(3*col_size, 1, nb_classes = 2)
	for n in range(0, trndata_tmp.getLength()):
		trndata.addSample( trndata_tmp.getSample(n)[0], trndata_tmp.getSample(n)[1] )
	## And this code converts 1 output to 40 binary outputs, to encode classes with one output neuron per class.
	trndata._convertToOneOfMany( )
	tstdata._convertToOneOfMany( )
	## Training.
	fnn = buildNetwork(trndata.indim, col_size , trndata.outdim, outclass=SoftmaxLayer)
	trainer = BackpropTrainer(fnn, dataset=trndata)
	## Compute percentage error.
	trainer.trainEpochs (n_epochs)
	score = (100 - percentError( trainer.testOnClassData (dataset=tstdata ), tstdata['class']))/100
	return score
