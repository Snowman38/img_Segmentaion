import os
import argparse
from matplotlib.pyplot import new_figure_manager
import torch
from tqdm import tqdm
from config import *
from dataset import get_data_loader
from utils import plot_training, save_model, save_losses, get_dice_score


def ev(val_loader, model, device, config):
    """Evaluates the model on the validation set.

    Args:
        val_loader (DatasetLoader): The loader for the validation set
        model (nn.Module): The model for the data
        device (torch.device): The device to run the step function on
        config (dict): The parameter dictionary
    """
    # Initialize the average loss
    mean_loss = 0

    for x, y in val_loader:
        # Ensure valid values
        config["criterion_info"]["name"] = "Dice"
        x, y = prepare_values(x, y, device, config)

        # Get the final activations
        output = model(x)

        # Append dice accuracy to the overall accuracy
        mean_loss += get_dice_score(output, y)

    # Calculate the actual average loss
    mean_loss /= len(val_loader.dataset)

    return mean_loss



def prepare_values(x, y, device, config):
    """Prepares the values for training/evaluation

    Args:
        x (torch.tensor): The input tensor
        y (torch.tensor): The target tensor
        device (torch.device): The device to run the step function on
        config (dict): The parameter dictionary
    """
    # Load the tensors to the proper device
    x, y = x.to(device), y.to(device, dtype=torch.long)

    # Ensure channel dimension of size `1` exists
    x.unsqueeze_(1)

    if config["model_info"]["input_channels"] > 1:
        # Expand channels if the model works on more than 1 channel
        x = x.repeat(1, config["model_info"]["input_channels"], 1, 1)

    # Loss function may expect or not expect the empty channel dimension
    y = prepare_for_loss(config["criterion_info"]["name"], y)

    return x, y


def step(data_loader, model, optimizer, criterion, device, config, eval=False):
    """Performs a step function for training/evaluating the model

    Args:
        data_loader (DataLoader): The dataset loader with x and y values
        model (nn.Module): The model for the data
        optimizer (optim.Optimizer): The optimizer
        criterion (nn.Module): The loss function
        device (torch.device): The device to run the step function on
        config (dict): The parameter dictionary
        eval (bool): Whether the model is being trained or evaluated
    
    Returns:
        (float): The total loss
    """
    if eval:
        # If `eval`, set the model to eval mode
        model.eval()
    else:
        # Otherwise set the model to train mode
        model.train()
    
    # Initialize the total loss
    total_loss = 0

    for x, y in data_loader:
        # Reset the gradient
        optimizer.zero_grad()
        
        # Ensure valid values
        x, y = prepare_values(x, y, device, config)

        # Get the final activations
        output = model(x)

        # Calculate the loss
        loss = criterion(output, y)

        if not eval:
            # Calculate the parameter gradients
            loss.backward()

            # Perform the parameter update
            optimizer.step()

        # Add the loss to the total loss
        total_loss += loss.item()

    return total_loss


def get_global_args():
    """Gets the arguments for the script file.

    Returns:
        (tuple): A tuple of parsed arguments including device and workers
    """
    # Initialize the parser
    parser = argparse.ArgumentParser()

    # Add available arguments to the parser
    parser.add_argument("--config", type=str, default=None,
                        help="config file path")
    parser.add_argument("--use_val", action="store_true", default=False,
                        help="use validation set to train")
    parser.add_argument("--save_loss", action="store_true", default=False,
                        help="save loss as npz")
    parser.add_argument("--save_checkpoints", action="store_true", default=False,
                        help="save checkpoints")
    
    # Parse the arguments
    args = parser.parse_args()

    if args.config is not None:
        # Load the config path
        config = load_config(args.config)
    else:
        # If config path is not specified
        raise ValueError("Invalid config path")
    
    # Get the correct device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"The current device is {device}")

    # Use 0 workers for Windows
    if os.name == "nt":
        num_workers = 0
    else:
        num_workers = 4
    
    return config, args.use_val, args.save_loss, args.save_checkpoints, device, num_workers


def get_config_args(config, device):
    """Gets the configuration arguments based on the config dictionary.

    Args:
        config (dict): The dictionary containig configuration files
        device (torch.device): The device the calculations are done on
    
    Returns:
        (tuple): A tuple of constructed arguments
    """
    # Get the model, optimizer and criterion from the config dictionary
    model = build_model(config['model_info']).to(device)
    optimizer = build_optimizer(model, config['optimizer_info'])
    criterion = build_criterion(config["criterion_info"]).to(device)

    
    # Get the training parameters from the config dictionary
    epochs = config["epochs"]
    batch_size = config["batch_size"]

    # Get the dataset info from the config dictionary
    train_path = config['dataset_info']["train_data_path"]
    val_path = config['dataset_info']["val_data_path"]
    

    return model, optimizer, criterion, epochs, batch_size, train_path, val_path


if __name__ == '__main__':
    # Get the global and the configutration arguments
    config, use_val, save_loss, save_checkpoints, device, num_workers = get_global_args()
    model, optimizer, criterion, epochs, batch_size, train_path, val_path = get_config_args(config, device)

    if use_val:
        # If full training is performed, use train and validation sets
        full_path = train_path, val_path
        train_loader = get_data_loader(full_path, batch_size, num_workers, full=True, n_aug=1)
    else:
        # Load training data and validation data
        n_aug = config["n_aug"]
        augment = True
        train_loader = get_data_loader(train_path, batch_size, num_workers, augment=augment, n_aug=n_aug)
        #train_acc_loader = get_data_loader(train_path, 1, num_workers, augment=False, shuffle=False)
        val_loader = get_data_loader(val_path, batch_size, num_workers, augment=False)
        #val_acc_loader = get_data_loader(val_path, 1, num_workers, augment=False, shuffle=False)

    # Initialize the list of train and validation losses
    train_losses, val_losses = [], []

    # Initialize loss values (for the tbar)
    train_loss = -1
    val_loss = -1
    acc = -1
    acc_tr = -1

    # Initialize the loading bar
    tbar = tqdm(range(epochs))
    tbar.set_description(f"[0] Train: N/A | Val: N/A")

    for epoch in tbar:
        # Calculate the training loss and append it to history
        train_loss = step(train_loader, model, optimizer, criterion, device, config)
        train_losses.append(train_loss)
        
        #if epoch % 10 == 0:
        #    acc_tr = ev(train_acc_loader, model, device, config)

        if not use_val:
            # If validation set is not used for training, record loss
            val_loss = step(val_loader, model, optimizer, criterion, device, config, eval=True)
            val_losses.append(val_loss)
            
            #if epoch % 10 == 0:
            #    acc = ev(val_acc_loader, model, device, config)
        
        if save_checkpoints and (epoch + 1) % 10 == 0:
            # If checkpoints are required, save model's checkpoints
            save_model(model, config, epoch + 1)
        
        # Update the description of the progress bar
        tbar.set_description(f"[{epoch+1}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | Acc tr {acc_tr:.4f} | Acc {acc:.4f}")

    # Plot the training curve
    plot_training(train_losses, val_losses)

    # Save the trained model
    save_model(model, config)

    if save_loss:
        # Save loss history if needed
        save_losses(train_losses, val_losses, config)
        
