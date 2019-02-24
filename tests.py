"""Main program"""
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import constants
from data_preprocessing import TestsetPreprocessing
from datasets import FacialPartsDataset
from model import CFPD
from trainer_and_tester import test_model

CONFIG_PATH = "./config.ini"

ORIGINAL_MODEL_PATH = "./data/models/trained_using_original_images.pth"
NORMALIZED_MODEL_PATH = "./data/models/trained_using_normalized_images.pth"
AUGMENTED_MODEL_PATH = "./data/models/trained_using_augmented_images.pth"


def main():
    parameters = constants.Parameters(CONFIG_PATH)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    commonset_preprocessing = TestsetPreprocessing(parameters, parameters.commonset_csv, parameters.commonset_dirs)
    commonset_preprocessing.process_images()

    df_commonset = pd.read_csv(parameters.commonset_csv)
    common_dataset = FacialPartsDataset(df_commonset, parameters.color, transform=data_transforms)

    challenging_preprocessing = TestsetPreprocessing(parameters, parameters.challengingset_csv,
                                                     parameters.challengingset_dirs)
    challenging_preprocessing.process_images()
    
    df_challengingset = pd.read_csv(parameters.challengingset_csv)
    challenging_dataset = FacialPartsDataset(df_challengingset, parameters.color, transform=data_transforms)

    w300_preprocessing = TestsetPreprocessing(parameters, parameters.w300set_csv, parameters.w300set_dirs)
    w300_preprocessing.process_images()
    
    df_w300set = pd.read_csv(parameters.w300set_csv)
    w300_dataset = FacialPartsDataset(df_w300set, parameters.color, transform=data_transforms)

    test_loaders = {
        "common_set": DataLoader(common_dataset, batch_size=32, num_workers=parameters.num_workers),
        "challenging_set": DataLoader(challenging_dataset, batch_size=32, num_workers=parameters.num_workers),
        "w300_set": DataLoader(w300_dataset, batch_size=32, num_workers=parameters.num_workers)
    }

    saved_model_paths = {
        "original": ORIGINAL_MODEL_PATH,
        "normalized": NORMALIZED_MODEL_PATH,
        "augmented": AUGMENTED_MODEL_PATH
    }

    checkpoints = {model_type: torch.load(model_path) for model_type, model_path in saved_model_paths.items()}
    device = torch.device("cuda:0" if torch.cuda.is_available() and parameters.gpu else "cpu")

    models = {model_type: CFPD(parameters.color) for model_type in saved_model_paths}

    for model_type, model in models.items():
        model.load_state_dict(checkpoints[model_type])
        model.to(device)

    loss_fn = torch.nn.MSELoss(reduction="elementwise_mean")

    output_path = "./data/tests/CFPD"

    for model_type, model in models.items():
        results_save_path = os.path.join(output_path, f"results_{model_type}.txt")
        test_model(model, test_loaders, loss_fn, results_save_path, device)


if __name__ == "__main__":
    main()
