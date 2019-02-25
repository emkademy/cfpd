"""
Main program
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

import constants
from datasets import FacialPartsDataset
from model import CFPD
from trainer_and_tester import train_model
from data_preprocessing import TrainsetPreprocessing

SEED = 1234
CONFIG_PATH = "./config.ini"

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    parameters = constants.Parameters(CONFIG_PATH)

    underscore_index = parameters.train_csv.find("_")
    train_base_csv = parameters.train_csv[:underscore_index] if underscore_index != -1 else parameters.train_csv
    train_base_csv += ".csv"
    trainset_preprocessor = TrainsetPreprocessing(parameters, train_base_csv, parameters.train_dirs)
    # Creates .csv files
    trainset_preprocessor.process_images()

    df = pd.read_csv(parameters.train_csv)
    df_train, df_validation = train_test_split(df, test_size=parameters.test_fraction, random_state=SEED)

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets = {
        "train": FacialPartsDataset(df_train, parameters.color, transform=data_transforms),
        "validation": FacialPartsDataset(df_validation, parameters.color, transform=data_transforms)
    }

    train_loader = DataLoader(image_datasets["train"], parameters.batch_size, parameters.shuffle,
                              num_workers=parameters.num_workers)
    validation_loader = DataLoader(image_datasets["validation"], parameters.batch_size, False,
                                   num_workers=parameters.num_workers)

    data_loaders = {
        "train": train_loader,
        "validation": validation_loader
    }

    dataset_lengths = {x: len(image_datasets[x]) for x in ["train", "validation"]}
    device = torch.device("cuda:0" if torch.cuda.is_available() and parameters.gpu else "cpu")

    model = CFPD(parameters.color)
    model.to(device)
    loss_fn = torch.nn.MSELoss(reduction="elementwise_mean")
    optimizer = optim.Adam(model.parameters(), lr=parameters.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=parameters.scheduler_step_size,
                                    gamma=parameters.scheduler_gamma)

    train_model(model, data_loaders, dataset_lengths, loss_fn, optimizer, scheduler, device, parameters.num_epochs,
                parameters.save_model_to, parameters.start_epoch)


if __name__ == "__main__":
    main()
