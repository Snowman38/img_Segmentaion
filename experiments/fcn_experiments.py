# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from matplotlib import pyplot as plt

from utils.config import build_model,load_config,build_optimizer
from utils.dataset import TrainDataset

def validation(data_loader, model, criterion, device):
    model.eval()
    test_loss = 0.0
    for i, sample in enumerate(data_loader):
        img, true_mask = sample
        img = img.to(device)
        true_mask = true_mask.to(device=device, dtype=torch.long)
        img.unsqueeze_(1)
        if input_channels > 1:
            img = img.repeat(1, input_channels, 1, 1)

        output = model(img)
        loss = criterion(output, true_mask)
        test_loss += loss.item()
#    end_for
    return test_loss


def training(data_loader, model, criterion, device):
    # training mode
    model.train()
    train_loss = 0.0
    # for each batch
    for i, sample in enumerate(data_loader):
        optimizer.zero_grad()
        # get batch data
        img, true_mask = sample
        # img and masks are [batch_size, 96, 96]
        img = img.to(device)
        true_mask = true_mask.to(device=device, dtype=torch.long)

        # Write your FORWARD below
        # Note: Input image to your model and ouput the predicted mask and Your predicted mask should have 4 channels
        # [batch_size, 96, 96] -> [batch_size, 1, 96, 96]. We only have grey scale, 1 is enough.
        img.unsqueeze_(1)
        if input_channels > 1:
            img = img.repeat(1, input_channels, 1, 1)
        output = model(img)

        # Then write your BACKWARD & OPTIMIZE below
        # Note: Compute Loss and Optimize
        loss = criterion(output, true_mask)
        loss.backward()
        optimizer.step()
        #         record train loss
        train_loss += loss.item()
    # end_for
    return train_loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None,
                        help='config file path')
    parser.add_argument('--use_val', action='store_true', default=False,
                        help='use validation set to train')
    parser.add_argument('--save_loss', action='store_true', default=False,
                        help='save loss as npz')
    parser.add_argument('--save_checkpoints', action='store_true', default=False,
                        help='save checkpoints')
    args = parser.parse_args()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"The current device is {device}")

    if args.config is not None:
        configs = load_config(args.config)
    else:
        print("invalid config path")
        sys.exit()
    input_channels = configs['model_info']['input_channels']
    # model = build_model(configs['model_info']).to(device)
    # optimizer = build_op(model,configs['optimizer_info'])

    batch_size = epochs = configs['batch_size']
    epochs = configs['epochs']
    if os.name == 'nt':  # Windows
        num_workers = 0
    else:  # Others
        num_workers = 4

    # load training data and validation data
    train_data_path = configs['dataset_info']["train_data_path"]
    train_set = TrainDataset(train_data_path)

    val_data_path = configs['dataset_info']["val_data_path"]
    val_set = TrainDataset(val_data_path)


    if args.use_val:
        train_dev_sets = torch.utils.data.ConcatDataset([train_set, val_set])
        training_data_loader = DataLoader(dataset=train_dev_sets, num_workers=num_workers, batch_size=batch_size,
                                          shuffle=True)
    else:
        training_data_loader = DataLoader(dataset=train_set, num_workers=num_workers, batch_size=batch_size,
                                          shuffle=True)
        val_data_loader = DataLoader(dataset=val_set, num_workers=num_workers, batch_size=batch_size, shuffle=True)

    fcn_list = ['FCNs','FCN8s','FCN16s','FCN32s']
    train_losses = np.zeros((len(fcn_list), epochs))
    val_losses = np.zeros((len(fcn_list), epochs))

    for i in range(len(fcn_list)):
        configs['model_info']["model_name"] = fcn_list[i]
        model = build_model(configs['model_info']).to(device)
        optimizer = build_optimizer(model, configs['optimizer_info'])
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(range(epochs))
        # epoch loop
        for epoch in tbar:
            train_loss = training(training_data_loader,model, criterion, device)
            train_losses[i,epoch] = train_loss
            tbar.set_description(f'Epoch {epoch + 1}, Train loss: {train_loss:.3f}')

            if not args.use_val:
                val_loss = validation(val_data_loader,model, criterion, device)
                val_losses[i,epoch] = val_loss
            if args.save_checkpoints and (epoch + 1) % 100 == 0:
                model_path = configs['model_info']['model_path']
                torch.save(model.state_dict(), f"{model_path}_ep{epoch+1}.pth")

    if args.save_loss:
        model_path = configs['model_info']['model_path']
        op_name = configs['optimizer_info']['name']
        with open(f"{model_path}_{op_name}_models_loss.npz","wb") as f:
            d = {'file': f, 'train_losses': train_losses, 'val_losses': val_losses,'model_list':fcn_list}
            np.savez(**d)

    # plot curve
    # plt.plot(range(len(train_losses)), train_losses, 'r', label="Training loss")
    # if not args.use_val:
    #     plt.plot(range(len(val_losses)), val_losses, 'g', label="Validation loss")
    # plt.xlabel("Training iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.ylim((0, 10))
    # plt.show()

    # save model
    # model_path = './models/unet.pth'
    # model_path = configs['model_info']['model_path']
    # torch.save(model.state_dict(), model_path)
