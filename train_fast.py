import model2 as model
import utilities
import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary
import os
import datetime
import numpy as np
import json
import h5py


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(train_folder, test_folder, model_file, cams, epochs=10, batch_size=40, learning_rate=0.0001, device=dev):
    # Create model and move to device
    pointmodel = model.PointNet().to(device)
    pcd_file_train = h5py.File('D:/MEMBEN/all_pcd_train.hdf5', 'r')
    pcd_file_test = h5py.File('D:/MEMBEN/all_pcd_test.hdf5', 'r')
    skeleton_train = h5py.File('D:/MEMBEN/skeleton_train.hdf5', 'r')
    skeleton_test = h5py.File('D:/MEMBEN/skeleton_test.hdf5', 'r')


    print("Device : " + str(dev))

    if os.path.exists(model_file):
        pointmodel.load_state_dict(torch.load(model_file))
        pointmodel.eval()

    # Create datasets and data loaders
    print("Creating Data sets and data loaders...")
    train_dataset = model.PointCloudDataset(path=train_folder, cams=cams, pcd_file=pcd_file_train, skeletons=skeleton_train)
    test_dataset = model.PointCloudDataset(path=test_folder, cams=cams,pcd_file=pcd_file_test, skeletons=skeleton_test)
    train_loader = model.torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = model.torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print("Data sets and data loaders created.")

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(pointmodel.parameters(), lr=learning_rate)

    train_loss = []
    val_loss = []

    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        print("Epoch " + str(epoch+1) + "/" + str(epochs))
        for i, data in enumerate(train_loader, 0):
            # Get inputs and move to device
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = pointmodel(inputs.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            train_loss.append(loss.item())
            if i % 110 == 0: # print every 10 mini-batches
                print('[', datetime.datetime.now(), ']', model_file, '[Epoch : %d, batch : %3d / %d] loss: %.3f' % (epoch + 1, i + 1, train_dataset.__len__()/batch_size, running_loss / 100))
                running_loss = 0.0

        # Save model
        if not os.path.exists('model/' + model_file):
            os.makedirs('model/' + model_file)
        torch.save(pointmodel.state_dict(), 'model/' + model_file + '/epoch' + str(epoch + 1))
        if not os.path.exists('train_loss/train_loss_' + model_file):
            os.makedirs('train_loss/train_loss_' + model_file)
        with open('train_loss/train_loss_' + model_file + '/epoch' + str(epoch + 1), "w") as fp:
            json.dump(np.asarray(train_loss).tolist(), fp)

        # Test loop
        with torch.no_grad():
            total_loss = 0.0
            for i, data in enumerate(test_loader, 0):
                # Get inputs and move to device
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Compute outputs and loss
                outputs = pointmodel(inputs.float())
                loss = criterion(outputs, labels.float())

                # Accumulate loss
                total_loss += loss.item()

            print('Test loss: %.3f' % (total_loss / len(test_loader)))
            val_loss.append(total_loss / len(test_loader))
    if not os.path.exists('val_loss'):
        os.makedirs('val_loss')
    with open('val_loss/val_loss_' + model_file, "w") as fp:
        json.dump(np.asarray(val_loss).tolist(), fp)
