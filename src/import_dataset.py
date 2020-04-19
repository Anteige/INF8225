import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.io import imread

def loadDataset(path):
    datasetX = []
    datasetY = []
    i = 0
    print("debut chargement image")
    for folder in os.listdir(path):
        for image in os.listdir(path + "/" + folder):
            if (image.endswith('.jpg')):
                X = imread(path + "/" + folder+ "/" + image)#[:,:,0]#[...,::-1]
                rows,cols,colors = X.shape
                X_size = rows*cols*colors
                X = X.reshape(X_size)
                datasetX.append(X)
                datasetY.append(folder.split("-")[1])
                if i == 100: 
                    #print("fin de la recuperation des images")
                    break
                i += 1

    return np.array(datasetX), np.array(datasetY)