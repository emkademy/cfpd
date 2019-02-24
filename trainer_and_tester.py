"""
The module provides a convenient function to train CFPD model given right parameters
"""
from collections import defaultdict
import copy
import os
import sys
import time

from matplotlib import pyplot as plt
import numpy as np
import torch

from utils import makedir


def train_model(model, data_loaders, dataset_sizes, loss_fn, optimizer, scheduler,
                device, num_epochs, model_save_path, start_epoch=0):
    losses = {"train": [], "validation": []}
    makedir(model_save_path)
    best_model = copy.deepcopy(model.state_dict())
    best_loss = sys.maxsize

    print_str = "Epoch {}/{} Phase: {} Batch: {}/{} Batch Loss: {} Time elapsed: {:.4f}m {:.4f}s"

    start = time.time()
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(60*"-")

        for phase in ["train", "validation"]:
            if phase == "train":
                for _ in range(start_epoch):
                    scheduler.step()
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            batch = 0
            for inputs, gt_coords in data_loaders[phase]:
                batch_start = time.time()
                inputs = inputs.to(device)
                gt_coords = gt_coords.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    output_coords = model(inputs)
                    loss = loss_fn(output_coords, gt_coords)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.shape[0]

                n_batches = dataset_sizes[phase]//inputs.shape[0]
                batch_end = time.time()
                batch_time_elapsed = batch_end - batch_start
                
                print(print_str.format(epoch, num_epochs, phase, batch, n_batches, loss.item(), 
                                       batch_time_elapsed//60, batch_time_elapsed % 60))
                batch += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            losses[phase].append(epoch_loss)

            print(f"Phase: {phase} Epoch: {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")

            if phase == "validation" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model, os.path.join(model_save_path, f"cfpd_model_{epoch}_{epoch_loss}.pth"))

        track_losses(losses, model_save_path)
    end = time.time()
    time_elapsed = end - start

    print(f"Training has been completed in {time_elapsed//60:.4f}m {time_elapsed%60:.4f}s")
    print(f"Minimum Loss: {best_loss:4f}")

    with open(os.path.join(model_save_path, "best_validation_loss.txt"), "w") as f:
        print(best_loss, file=f)


def track_losses(losses, save_path):
    validation_losses = losses["validation"]
    train_losses = losses["train"]

    save_txt = np.column_stack((range(len(validation_losses)), train_losses, validation_losses))
    save_losses = os.path.join(save_path, "losses.txt")
    np.savetxt(save_losses, save_txt)

    draw_loss_graph(train_losses, validation_losses, save_path)


def draw_loss_graph(train_losses, validation_losses, save_path):
    plt.plot(validation_losses, label="Validation Losses")
    plt.plot(train_losses, label="Train Losses")
    plt.legend(loc='upper right')
    plt.ylim(top=np.max([validation_losses[0], train_losses[0]]))
    save_path = os.path.join(save_path, "loss_graph.jpg")
    plt.savefig(save_path)
    plt.clf()


def test_model(model, test_loaders, loss_fn, results_save_path, device, best_loss=None):
    """
    Test model
    """
    dataset_losses = defaultdict(lambda: [])
    model.eval()
    makedir(os.path.dirname(results_save_path))
    results = ""
    if best_loss:
        results += f"Validation lost: {best_loss}\n"
    for testset_name in test_loaders:
        running_loss = 0.0
        with torch.no_grad():
            for inputs, gt_coords in test_loaders[testset_name]:
                inputs = inputs.to(device)
                gt_coords = gt_coords.to(device)

                output_coords = model(inputs)
                loss = loss_fn(output_coords, gt_coords)
                dataset_losses[testset_name].append(loss.item())

                running_loss += loss.item() * inputs.shape[0]

        epoch_loss = running_loss / len(test_loaders[testset_name].dataset)
        print_str = f"{testset_name} Loss: {epoch_loss:.4f}\n"
        results += print_str
        print(print_str)

    dataset_losses["full_set"] = dataset_losses["common_set"] + dataset_losses["challenging_set"]
    full_set_loss = np.mean(dataset_losses["full_set"])
    results += f"full_set loss: {full_set_loss}\n"

    with open(results_save_path, "w") as file_:
        print(results, file=file_)

    return dataset_losses
