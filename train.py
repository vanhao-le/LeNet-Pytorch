from model.LeNet import LeNet
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import random_split
from torchvision.datasets import KMNIST
import torchvision.transforms as transforms
from chip import config
import torch
import torch.nn as nn

def main():
    tranform_lst = transforms.Compose([
        transforms.ToTensor()
    ])
    # load the KMNIST dataset
    print("[INFO] loading the KMNIST dataset...")
    k_dataset = KMNIST(root="data", train=True, download=True, transform=tranform_lst)    
    # calculate the train/validation split
    print("[INFO] generating the train/validation split...")
    dataset_size = len(k_dataset)
    num_trains = int(dataset_size * config.TRAIN_SPLIT)
    num_vals = dataset_size - num_trains

    (train_ds, val_ds) = random_split(
        k_dataset, 
        [num_trains, num_vals], 
        generator=torch.Generator().manual_seed(42)
    )
    # 45000 samples for training, 15000 samples for validation
    # 10000 samples for testing
    # print(len(train_ds), len(val_ds), len(test_ds))

    # initialize the train, validation, and test data loaders
    # set shuffle for only training purpose 
    train_loader = DataLoader(train_ds, shuffle=True, batch_size = config.BATCH_SIZE)
    val_loader = DataLoader(val_ds, batch_size = config.BATCH_SIZE)
    

    # calculate steps per epoch for training and validation set
    train_steps = len(train_loader.dataset) // config.BATCH_SIZE
    val_steps = len(val_loader.dataset) // config.BATCH_SIZE

    num_classes = len(train_ds.dataset.classes)

    # initialize the LeNet model
    print("[INFO] initializing the LeNet model...")
    model = LeNet(input_dim= 1, num_classes=num_classes)
    print(model)
    model.to(config.DEVICE)
    
    # initialize our optimizer and loss function
    opt = optim.Adam(model.parameters(), lr=config.INIT_LR)
    loss_func = nn.CrossEntropyLoss()
    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    # measure how long training is going to take
    print("[INFO] training the network...")
    # loop over our epochs
    for e in range(config.EPOCHS):
        # set model for training
        model.train()
        # initialize loss values for training / validation
        total_train_loss = 0.0
        total_val_loss = 0.0
        # initialize corrected predictions for training / validation
        train_correct = 0
        val_correct = 0
        # loop over training set
        for (imgs, lbs) in train_loader:
            # send input to gpu
            # print(imgs.shape, lbs.shape)
            (imgs, lbs) = (imgs.to(config.DEVICE) , lbs.to(config.DEVICE))
            # perform forward()
            preds = model(imgs)
            loss = loss_func(preds, lbs)
            # zero gradient 
            opt.zero_grad()
            # perform backpropagation step
            loss.backward()
            opt.step()
            # add the loss to the total training loss
            total_train_loss += loss
            train_correct += (preds.argmax(1) ==  lbs).type(torch.float).sum().item()

        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (imgs, lbs) in val_loader:
                # send input to gpu
                (imgs, lbs) = (imgs.to(config.DEVICE) , lbs.to(config.DEVICE))
                # make the predictions and calculate the validation loss
                preds = model(imgs)
                loss = loss_func(preds, lbs)
                total_val_loss += loss
                # calculate the number of correct predictions
                val_correct += (preds.argmax(1) ==  lbs).type(torch.float).sum().item()
        
        
        # calculate the average training and validation loss
        avg_train_loss = total_train_loss / train_steps
        avg_val_loss = total_val_loss / val_steps
        # calculate the training and validation accuracy
        train_acc = train_correct / len(train_ds)
        val_acc = val_correct / len(val_ds)

        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(e + 1, config.EPOCHS))
        print("Train loss: {:.6f}, Train accuracy: {:.2f}".format(avg_train_loss, train_acc))
        print("Val loss: {:.6f}, Val accuracy: {:.2f}".format(avg_val_loss, val_acc))

        
        # cpu_avg_train_loss = torch.tensor(avg_train_loss, dtype=torch.float32)
        # cpu_avg_val_loss = torch.tensor(avg_val_loss, dtype=torch.float32)
        cpu_avg_train_loss = avg_train_loss.clone().detach()
        cpu_avg_val_loss = avg_val_loss.clone().detach()
        # update our training history
        H["train_loss"].append(cpu_avg_train_loss.clone().detach().cpu().numpy())
        H["train_acc"].append(train_acc)
        H["val_loss"].append(cpu_avg_val_loss.clone().detach().cpu().numpy())
        H["val_acc"].append(val_acc)

    
    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H["train_loss"], label="train_loss")
    plt.plot(H["val_loss"], label="val_loss")
    plt.plot(H["train_acc"], label="train_acc")
    plt.plot(H["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(config.MODEL_PLOT)

    # serialize the model to disk
    torch.save(model, config.MODEL_PATH)

    return
if __name__ == '__main__':
    main()
    