import torch
import sys
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from training_exp_safe_mask import Dynamic
from train_dynamics import StateTransitionDataset
import math


def stat_eval(dynamic_file, data_file):

    data = torch.load(data_file)

    inputs = data["inputs"]
    outputs = data["outputs"]

    dataset = StateTransitionDataset(inputs, outputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = Dynamic(dynamic_file)

    model.float()

    total_mse = 0.0
    total_mae = 0.0
    n_samples = 0

    with torch.no_grad():
        for inputs_batch, targets_batch in loader:
            preds = model(inputs_batch.float())

            # Mean squared error and mean absolute error
            mse = F.mse_loss(preds, targets_batch)#, reduction='sum')
            mae = F.l1_loss(preds, targets_batch)#, reduction='sum')

            total_mse += mse.item() * inputs_batch.size(0)
            total_mae += mae.item() * inputs_batch.size(0)
            n_samples += inputs_batch.size(0)

    avg_mse = total_mse / n_samples
    avg_mae = total_mae / n_samples

    return avg_mse, avg_mae


def make_graphs(dynamic_file, data_file, graph_out_name):

    data = torch.load(data_file)

    inputs = data["inputs"]  
    outputs = data["outputs"]

    model = Dynamic(dynamic_file)

    model.eval()

    # Get predictions
    with torch.no_grad():
        preds = model(inputs[:100])

    # Convert to numpy
    preds = preds.numpy()
    true_outputs = outputs[:100].detach().numpy()

    # Plot each output dimension
    labels = ['x\'', 'y\'', 'vx\'', 'vy\'']

    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()  # flatten 2D array of axes into 1D list

    for i in range(4):
        ax = axes[i]
        ax.scatter(range(len(true_outputs)), true_outputs[:, i], label='True', s=25, alpha=0.6)
        ax.scatter(range(len(preds)), preds[:, i], label='Predicted', s=15, alpha=0.6)
        ax.set_title(f"{labels[i]} (Predicted vs True)")
        ax.set_xlabel("Sample index")
        ax.set_ylabel(labels[i])
        ax.grid(True)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    plt.savefig(f"{graph_out_name}.png", dpi=300)
    print(f"Saved plot to {graph_out_name}.png")
    plt.show()

    # Compute absolute errors
    abs_errors = np.abs(preds - true_outputs)
    labels = ['x\'', 'y\'', 'vx\'', 'vy\'']

    # Plot absolute errors (no connecting lines, larger dots)
    plt.figure(figsize=(10, 6))
    x = np.arange(len(abs_errors))

    for i in range(4):
        plt.scatter(x, abs_errors[:, i], s=30, label=f"Error in {labels[i]}", alpha=0.7)

    plt.xlabel("Sample index")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Prediction Errors per Output Dimension")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{graph_out_name}_absolute_errors_scatter.png", dpi=300)
    print(f"Saved plot to {graph_out_name}_absolute_errors_scatter.png")
    plt.show()

def get_lip_bound(dynamic_file):
        
        model = Dynamic(dynamic_file)

        model.eval()

        lip_product = 1.0
        
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                W = layer.weight
                sigma = torch.linalg.norm(W, ord=2)  # Spectral norm
                lip_product *= sigma

        Linf = False

        if Linf:
            lip_product = math.sqrt(6) * lip_product / 2.0

        return lip_product

def find_max_error(dynamic_file, data_file):

    data = torch.load(data_file)

    inputs = data["inputs"].float()
    outputs = data["outputs"].float()

    model = Dynamic(dynamic_file)

    model.eval()

    max_error = 0.0

    with torch.no_grad():
        for input, output in zip(inputs, outputs):
            pred = model(input)

            error = pred - output
            
            linf = torch.norm(error, p=float('inf'))

            if linf > max_error:
                max_error = linf
            

    print(f"\nMax error: {max_error}\n")

    return max_error

def find_max_error_grid(dynamic_file, spacing):
    pos_limit = 2.2
    vel_limit = 0.55
    force_limit = 1.5

    rows = []

    for x in np.arange(-pos_limit, pos_limit, spacing):
        for y in np.arange(-pos_limit, pos_limit, spacing):
            for vx in np.arange(-vel_limit, vel_limit, spacing):
                for vy in np.arange(-vel_limit, vel_limit, spacing):
                    for fx in np.arange(-force_limit, force_limit, spacing):
                        for fy in np.arange(-force_limit, force_limit, spacing):
                            point = torch.tensor([x, y, vx, vy, fx, fy], dtype=torch.float64)
                            rows.append(point)

    x_train = torch.stack(rows)

    print(x_train.shape)

    outputs_train = next_state_batch



if __name__ == "__main__":

    dynamic = "dynamics/dynamic_0.pt"
    data_eval = "dynamic_data/train_0.pt"

    # pred_out_name = dynamic.split(".")[0]
    # split_out_name = data_eval.split(".")[0]
    # graph_out_name = f"{pred_out_name}_{split_out_name}"

    mse, mae = stat_eval(dynamic, data_eval)
    print(f"Avg MSE: {mse}")
    print(f"Avg MAE: {mae}")

    #make_graphs(dynamic, data_eval, graph_out_name)