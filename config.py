import json
from torch import optim
from torch import nn
from architectures import *
from monai.losses import *
from utils import AdaHessian


def get_generic_config():
    """This method gets json configuration file for any generic net.
    
    Returns:
        (dict): A dictionary containing model parameters
    """
    # Describe the default configuration
    config = {
        "epochs": 100,
        "batch_size": 32,
        "dataset_info": {
            "train_data_path": "data/train",
            "val_data_path": "data/val",
            "test_data_path": "data/test"
        },
        "model_info": {
            "name": "MyModel",
            "input_channels": 1,
            "model_path": "models/my_model.pth"
        },
        "optimizer_info": {
            "name": "MyOptimizer",
            "lr": .001,
            "weight_decay": 1e-8,
        },
        "criterion_info": {
            "name": "MyCriterion",
        }
    }

    return config


def load_config(path):
    """Loads model configuration.
    
    Returns:
        (dict): A dictionary containing model parameters
    """
    # Extract data from json file
    with open(path) as json_file:
        config = json.load(json_file)

    return config


def build_model(model_info):
    """Builds a model based on the provided parameters.

    Args:
        model_info (dict): The model parameters

    Returns:
        (nn.Module): A model built on the provided specifications
    """
    if model_info["name"] == "FCN":
        # Generate FCN model
        model = FCN8s(
            pretrained_net=VGGNet(requires_grad=True),
            n_class=4
        )
    elif model_info["name"] == "UNet":
        # Generate UNet model
        model = UNet(
            in_channels=model_info["input_channels"],
            depth=model_info["depths"],
            num_classes=4
        )
    elif model_info["name"] == "SegNet":
        # Generate SegNet model
        model = SegNet(
            input_channels=model_info["input_channels"],
            output_channels=4,
        )
    elif model_info["name"] == "DeepLab_v3":
        # Generate DeepLab model
        model = DeepLab(
            backbone="drn",
            output_stride=16,
            num_classes=4
        )
    elif model_info["name"] == "EffUNet":
        # Generate EffUNet model
        model = get_efficientunet_b0(
            out_channels=4,
            concat_input=True,
            pretrained=True
        )
    elif model_info["name"] == "Swin":
        # Generate Swin Transformer model
        model = SwinTransformerSys(
            img_size=96,
            in_chans=1,
            hidden_dim=96,
            num_classes=4,
            window_size=6,
            drop_rate=0
        )
    elif model_info["name"] == "CSwin":
        # Generate CSwin Transformer model
        model = CSWinTransformer(
            img_size=96,
            in_chans=1,
            num_classes=4
        )
    elif model_info["name"] == "Segmenter":
        # Generate Segmenter Transformer model
        model = Segmenter(
            backbone="vit_base_patch16_224",
            num_classes=4,
            image_size=96,
            emb_dim=96,
            hidden_dim=96,
            num_layers=3,
            num_heads=12,
        )
    elif model_info["name"] == "SETR":
        # Generate SETR model
        model = SETRModel(
            in_channels=1,
            out_channels=4
        )
    elif model_info["name"] == "Ensemble":
        # Generate model ensemble
        model = build_ensamble(get_configs())
    else:
        raise ValueError(f"Model {model_info['name']} is invalid!")
    
    return model


def build_optimizer(model, optimizer_info):
    """Builds an optimizer based on the provided parameters.

    Args:
        model (nn.Module): The model whose parameters to update
        model_info (dict): The optimizer parameters

    Returns:
        (optim) An optimizer built on the provided specifications
    """
    if optimizer_info["name"] == "AdaHessian":
        # Generate AdaHessian optimizer
        optimizer = AdaHessian(
            model.parameters(),
            lr=optimizer_info["lr"],
            weight_decay=optimizer_info["weight_decay"]
        )
    elif optimizer_info["name"] == "Adam":
        # Generate Adam optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=optimizer_info["lr"],
            weight_decay=optimizer_info["weight_decay"]
        )
    elif optimizer_info["name"] == "RMSprop":
        # Generate RMSProp optimizer
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=optimizer_info["lr"],
            weight_decay=optimizer_info["weight_decay"],
            momentum=optimizer_info["momentum"]
        )
    elif optimizer_info["name"] == "SGD":
        # Generate SGD optimizer
        optimizer = optim.SGD(
            model.parameters(),
            lr=optimizer_info["lr"],
            momentum=optimizer_info["momentum"],
            weight_decay=optimizer_info["weight_decay"],
            nesterov=True
        )
    else:
        raise ValueError(f"Optimizer {optimizer_info['name']} is invalid!")

    return optimizer


def build_criterion(criterion_info):
    """Builds a criterion based on the provided parameters.

    Args:
        criterion_info (dict): The loss parameters

    Returns:
        (nn.Module): A criterion built on the provided specifications
    """
    if criterion_info["name"] == "CrossEntropy":
        # Generate Cross Entropy criterion
        criterion = nn.CrossEntropyLoss()
    elif criterion_info["name"] == "Dice":
        # Generate Dice criterion
        criterion = DiceLoss(to_onehot_y=True, softmax=True, jaccard=True)
    elif criterion_info["name"] == "DiceCE":
        # Generate Dice Cross Entropy criterion
        criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    elif criterion_info["name"] == "DiceFocal":
        # Generate Dice Focal criterion
        criterion = DiceFocalLoss(to_onehot_y=True, softmax=True)
    elif criterion_info["name"] == "Focal":
        # Generate Focal criterion
        criterion = FocalLoss(to_onehot_y=True)
    elif criterion_info["name"] == "Tversky":
        # Generate Tversky criterion
        criterion = TverskyLoss()
    elif criterion_info["name"] == "FocalTversky":
        # Generate Cross Entropy criterion
        raise NotImplementedError("Focal Tversky is not yet implemented!")
    elif criterion_info["name"] == "Contrastive":
        # Generate Contrastive criterion
        criterion = ContrastiveLoss()
    else:
        raise ValueError(f"Optimizer {criterion_info['name']} is invalid!")
    
    return criterion


def prepare_for_loss(loss_name, target):
    """Prepares the target for the loss function.
    
    Args:
        loss_name (str): The name of the loss funciton
        target (torch.tensor): The target values of shape (N, H, W)
    
    Returns:
        (torch.tensor): The prepared target tensor
    """
    # Loss functions that require channel dimension
    REQUIRES_CHANNEL_DIM = [
        "Dice",
        "DiceCE",
        "DiceFocal",
        "Focal",
        "Tversky",
        "FocalTversky",
        "Contrastive"
    ]

    if loss_name in REQUIRES_CHANNEL_DIM:
        # Append extra channel dimension
        target = target.unsqueeze_(1)
    
    return target



def get_configs():
    paths = [
        'configs\effunet_full_100_dice.json',
        'configs\effunet_full_150_dice.json',
        'configs\effunet_full_150_dicefocal.json',
        "configs\deeplab_drn.json",
        #"configs\deeplab_resnet.json"
    ]
    
    configs = [load_config(path) for path in paths]
    
    return configs




def build_ensamble(configs):
    """Creates the model ensamble of n different models.

    Args:
        infos (list): List of model configurations

    Returns:
        (nn.Module): A model ensemble (segmentation voter)
    """
    # Check the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize models and channels
    ms = []
    cs = []

    for config in configs:
        # Load the trained model
        m = build_model(config["model_info"]).to(device)
        m.load_state_dict(torch.load(config["model_info"]["model_path"]))

        # Append the model and the channel to the list
        ms.append(m)
        cs.append(config["model_info"]["input_channels"])

    return VotingSegmenter(ms, cs)


def create_ensamble():
    """Creates the model ensamble of 5 different models.

    Returns:
        (nn.Module): A model ensemble (segmentation voter)
    """
    # Check the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the configuration files
    config1 = load_config("configs/fcn.json")
    config2 = load_config("configs/unet.json")
    config3 = load_config("configs/segnet.json")
    config4 = load_config("configs/deeplab.json")
    config5 = load_config("configs/effunet.json")

    # Get the number of channels for each model
    c1 = config1["model_info"]["input_channels"]
    c2 = config2["model_info"]["input_channels"]
    c3 = config3["model_info"]["input_channels"]
    c4 = config4["model_info"]["input_channels"]
    c5 = config5["model_info"]["input_channels"]

    # Build 5 different models
    m1 = build_model(config1["model_info"]).to(device)
    m2 = build_model(config2["model_info"]).to(device)
    m3 = build_model(config3["model_info"]).to(device)
    m4 = build_model(config4["model_info"]).to(device)
    m5 = build_model(config5["model_info"]).to(device)

    # Load the model parameters
    m1.load_state_dict(torch.load(config1["model_info"]["model_path"]))
    m2.load_state_dict(torch.load(config2["model_info"]["model_path"]))
    m3.load_state_dict(torch.load(config3["model_info"]["model_path"]))
    m4.load_state_dict(torch.load(config4["model_info"]["model_path"]))
    m5.load_state_dict(torch.load(config5["model_info"]["model_path"]))

    return VotingSegmenter(m1, m2, m3, m4, m5, c1, c2, c3, c4, c5)


if __name__ == '__main__':
    # Get generic config data
    data = get_generic_config()

    # Save the configuration file to `configs` dir
    with open(f"configs/generic.json", 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)
    