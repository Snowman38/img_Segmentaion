import os
import cv2
import torch
import numpy as np
from dataset import get_data_loader
from utils import get_dice_score
from train import get_global_args, get_config_args, prepare_values

ENSAMBLE = True

def rle_encoding(x):
    '''
    *** Credit to https://www.kaggle.com/rakhlin/fast-run-length-encoding-python ***
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def submission_converter(mask_directory, path_to_save):
    writer = open(path_to_save, 'w')
    writer.write('id,encoding\n')

    files = os.listdir(mask_directory)

    for file in files:
        name = file[:-4]
        mask = cv2.imread(os.path.join(mask_directory, file), cv2.IMREAD_UNCHANGED)

        mask1 = (mask == 1)
        mask2 = (mask == 2)
        mask3 = (mask == 3)

        encoded_mask1 = rle_encoding(mask1)
        encoded_mask1 = ' '.join(str(e) for e in encoded_mask1)
        encoded_mask2 = rle_encoding(mask2)
        encoded_mask2 = ' '.join(str(e) for e in encoded_mask2)
        encoded_mask3 = rle_encoding(mask3)
        encoded_mask3 = ' '.join(str(e) for e in encoded_mask3)

        writer.write(name + '1,' + encoded_mask1 + "\n")
        writer.write(name + '2,' + encoded_mask2 + "\n")
        writer.write(name + '3,' + encoded_mask3 + "\n")

    writer.close()


def evaluate(val_loader, model, device, config):
    """Evaluates the model on the validation set.

    Args:
        val_loader (DatasetLoader): The loader for the validation set
        model (nn.Module): The model for the data
        device (torch.device): The device to run the step function on
        config (dict): The parameter dictionary
    """
    
    #print Info
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    n_aug = config["n_aug"]
    augment = config["augment"]
    name = config['model_info']["name"]
    name1 = config["criterion_info"] ["name"]
    name2 = config['optimizer_info']
    print(f"batch_size:{batch_size}  epochs:{epochs}  n_aug:{n_aug}  augment:{augment}  Models:{name} Loss:{name1} optimizer:{name2}")


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
        print(get_dice_score(output, y))
        

    # Calculate the actual average loss
    mean_loss /= len(val_loader.dataset)
    
    # Print the average loss
    print(f"Average dice loss on the validation set = {mean_loss}")


def predict(test_loader, model, device, config):
    """Generates the predictions for the test set

    Args:
        val_loader (DatasetLoader): The loader for the validation set
        model (nn.Module): The model for the data
        device (torch.device): The device to run the step function on
        config (dict): The parameter dictionary
    """
    for i, x in enumerate(test_loader):
        # Get the valid input parameters
        x = x.to(device)
        x.unsqueeze_(1)
        x = x.repeat(1, config["model_info"]["input_channels"], 1, 1)

        # Generate the prediction
        output = model(x)
        _, pred = torch.max(output, 1)

        # Save the predicted masks to `test/mask` dir
        test_mask_path = config["dataset_info"]["test_data_path"] + "/mask"
        pred_mask_path = os.path.join(test_mask_path, f"cmr{121 + i}_mask.png")
        cv2.imwrite(pred_mask_path, pred.cpu()[0, ...].numpy())


if __name__ == '__main__':
    # Get the global and the configutration arguments
    config, use_val, _, _, device, num_workers = get_global_args()
    model, _, _, _, _, _, val_path = get_config_args(config, device)

    if not ENSAMBLE:
        # Load weights to the model if it's not ensamble
        model_path = config["model_info"]["model_path"]
        model.load_state_dict(torch.load(model_path))
    
    # Set model's state to `eval`
    model.eval()

    if use_val:
        # Load test data with batch size `1` and generate results
        test_path = config["dataset_info"]["test_data_path"]
        test_loader = get_data_loader(test_path, 1, num_workers, shuffle=False, mask_exists=False, augment=False, full=True)
        predict(test_loader, model, device, config)

        # Save the predictions to submission file
        pred_path = os.path.join(test_path, "mask")
        subm_path = config["model_info"]["name"] + "_submission.csv"
        submission_converter(pred_path, subm_path)
        print("Submission file saved.")
    else:
        # Load val data with batch size `1` and print results
        val_loader = get_data_loader(val_path, 1, num_workers, augment=False, shuffle=False)
        evaluate(val_loader, model, device, config)


