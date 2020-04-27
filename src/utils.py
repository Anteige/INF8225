import os
import torch
import numpy as np
from skimage.io import imread
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

def load_dataset(path):
    datasetX = []
    datasetY = []
    i = 0
    print("debut chargement image")
    for folder in os.listdir(path):
        for image in os.listdir(path + "/" + folder):
            if (image.endswith('.jpg')):
                X = imread(path + "/" + folder+ "/" + image)
                X = X.reshape(3, 224, 224)
                datasetX.append(X)
                datasetY.append(folder.split("-")[1])

    return np.array(datasetX), np.array(datasetY)   

def train(n_epoch, model, train_loader, lr, weight_decay, device):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()
    train_losses = []
    for epoch in range(n_epoch):
        model.train()
        tr_loss = 0
        for i, data in enumerate(train_loader, 0):   
            if device.type == 'cuda':         
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = data

            optimizer.zero_grad()
            output_train = model(inputs)
            loss_train = criterion(output_train, labels.long())
            train_losses.append(loss_train)
            loss_train.backward()
            optimizer.step()
            tr_loss += loss_train.item()

        if epoch%2 == 0:
            print('Epoch : ',epoch+1, '\t', 'loss :', loss_train)
        del loss_train, output_train, inputs, labels
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return train_losses

def test(model, test_loader, device, top_10=False):
    correct = 0
    total = 0
    with torch.no_grad():
        i = 0
        for data in test_loader:    
            if device.type == 'cuda':
                images, labels = data[0].to(device), data[1].to(device)
            else:
                images, labels = data
            outputs = model(images)
            if top_10:
                _, predicted = torch.topk(outputs.data, 10)
                if labels in predicted:
                    correct += 1
            else:
                _, predicted = torch.max(outputs.data, 1)
                if predicted == labels: 
                    correct += 1
            total += labels.size(0)
            del images, labels
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    return 100 * correct / total
