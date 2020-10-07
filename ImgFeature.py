# -*- coding: utf-8 -*-


from keras.layers import GlobalAveragePooling2D, Input, Lambda, AveragePooling1D
import cv2 #Computer Vision library
import numpy as np
import pandas as pd
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
tqdm.pandas()

import scipy
import sklearn
import keras
#import torch
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from keras.models import Model
import multiprocessing #Might not work on all computers be careful
from keras.applications.densenet import preprocess_input, DenseNet121 #transfer learning 121 and imput preprocessor
trainData = pd.read_csv("train.csv")
testData = pd.read_csv("test/test.csv")
trainMetaPath = "train_metadata/"
testMetaPath = "test_metadata/"
trainData.head()

#converts all images to squares(easier for CNN input to have uniform shape and size)
def imgToSqr(img, size):
    ratio = float(size) / max(img.shape[:2])
    newSize = tuple([int(x * ratio) for x in img.shape[:2]])
    img = cv2.resize(img, (newSize[1], newSize[0]))
    dw = size - newSize[1]#difference in width
    dh = size - newSize[0]#difference in height
    top, bottom = dh // 2, dh - (dh // 2)#defining image borders
    left, right = dw // 2, dw - (dw // 2)
    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img

def loadImage(path, pet_id, size, toFlip=False):
    img = cv2.imread(f'{path}{pet_id}-1.jpg') #only using 1st image for every pet
        
    if toFlip:
        img = cv2.flip(img, 1)
    
    new_img = imgToSqr(img, size)
    new_img = preprocess_input(new_img) #converts image size to 512
    return new_img

def featureExtractor():
    imgSize=512
    batches=16
    inputLayer = Input((imgSize,imgSize,3))
    #transfer learning
    transferPath= "DenseNet-BC-121-32-no-top.h5"
    trainImgPath="train_images/"
    testImgPath="test_images/"
    denseNetwork = DenseNet121(input_tensor=inputLayer, weights=transferPath, include_top=False)
    x = denseNetwork.output
    networkOutput = GlobalAveragePooling2D()(x)
    networkModel=Model(inputLayer,networkOutput)
    featureDimensions = int(networkModel.output.shape[1])
    
    trainFeatures = np.zeros((len(trainData), featureDimensions))
    trainFlippedFeatures = np.zeros((len(trainData), featureDimensions))
    testFeatures = np.zeros((len(testData), featureDimensions))
    

    
    for dataFrame, path, features, arguments in [
        (trainData, trainImgPath, trainFeatures, {}),
        (trainData, trainImgPath, trainFlippedFeatures, {'flip': True}),
        (testData, testImgPath, testFeatures, {})]:
        
        pet_ids = dataFrame['PetID'].values
        noBatches = int(np.ceil(len(pet_ids) / batches))
        
        for batch in tqdm(range(noBatches)):
            print(batch/noBatches)
            startBatch = batch*batches
            endBatch = (batch+1)*batches
            batch_pets = pet_ids[startBatch:endBatch]
            batch_images = np.zeros((len(batch_pets), imgSize, imgSize, 3))
            for i,pet_id in enumerate(batch_pets):
                try:
                    batch_images[i] = loadImage(path, pet_id, imgSize, **arguments)
                except:
                    pass
            batch_preds = networkModel.predict(batch_images)
            for i,pet_id in enumerate(batch_pets):
                features[batch * batches + i] = batch_preds[i]
                
    np.save('img_features_train.npy', trainFeatures)
    np.save('img_features_train_flipped.npy', trainFlippedFeatures)
    np.save('img_features_test.npy', testFeatures)    
   

process = multiprocessing.Process(target=featureExtractor)#parallelization
process.start()
process.join()#concatenating processes

featureExtractor()

from keras import backend as k
k.tensorflow_backend._get_available_gpus()

trainFeatures = np.load('img_features_train.npy')
trainFeaturesFlipped = np.load('img_features_train_flipped.npy')
testFeatures = np.load('img_features_test.npy')