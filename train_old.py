import model
import utilities
import torch
import torch.nn as nn
import torch.optim as optim
import torchsummary
import os
import datetime
import numpy as np
import json


dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(train_folder, test_folder, model_file, cams, epochs=10, batch_size=80, learning_rate=0.0001, device=dev):
    # Create model and move to device
    pointmodel = model.PointNet().to(device)
    torchsummary.summary(pointmodel, (3, 5000))

    print("Device : " + str(dev))

    if os.path.exists(model_file):
        pointmodel.load_state_dict(torch.load(model_file))
        pointmodel.eval()

    # Create datasets and data loaders
    print("Creating Data sets and data loaders...")
    train_dataset = model.PointCloudDataset(path=train_folder, cams=cams)
    test_dataset = model.PointCloudDataset(path=test_folder, cams=cams)
    train_loader = model.torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = model.torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print("Data sets and data loaders created.")

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(pointmodel.parameters(), lr=learning_rate)

    save_loss = []

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
            save_loss.append(loss.item())
            if i % 10 == 0: # print every 10 mini-batches
                print('[', datetime.datetime.now(), ']', model_file, '[Epoch : %d, batch : %3d / %d] loss: %.3f' % (epoch + 1, i + 1, train_dataset.__len__()/batch_size, running_loss / 100))
                running_loss = 0.0

        # Save model
        torch.save(pointmodel.state_dict(), model_file + '_epoch' + str(epoch))
        with open('train_loss/train_loss_' + model_file + '_epoch' + str(epoch), "w") as fp:
            json.dump(np.asarray(save_loss).tolist(), fp)

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
